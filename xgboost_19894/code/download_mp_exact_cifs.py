#!/usr/bin/env python3
"""Download exact-match CIFs from Materials Project for missing/3DSC formulas.

The script reads:
  - missing_cif_formula_summary_after_external.csv
  - 02_exact_plus_3dsc/exact_plus_3dsc_3dsc_approx_cif_inventory.csv

It queries Materials Project by chemical system, filters returned materials by
stoichiometric equality to the target formulas, then writes simple P1 CIF files
from the returned structure JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import requests


MP_SUMMARY_URL = "https://api.materialsproject.org/materials/summary/"
SUMMARY_FIELDS = (
    "material_id,formula_pretty,formula_anonymous,energy_above_hull,"
    "is_stable,theoretical,deprecated,formation_energy_per_atom,nsites"
)
STRUCTURE_FIELDS = SUMMARY_FIELDS + ",structure"


class FormulaError(ValueError):
    pass


@dataclass
class TargetFormula:
    target_id: str
    formula: str
    formula_reduced: str
    canonical_key: str
    chemsys: str
    composition: OrderedDict[str, Fraction]
    source_categories: str
    source_rows: int
    ml_rows: int
    source_records: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download exact stoichiometric CIF matches from Materials Project."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Dataset base directory containing the CSV inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mp_exact_cif_pull"),
        help="Output directory, relative to base-dir unless absolute.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MP_API_KEY", ""),
        help="Materials Project API key. Prefer setting MP_API_KEY instead.",
    )
    parser.add_argument(
        "--match-tol",
        type=float,
        default=1e-4,
        help="Absolute tolerance on normalized atomic fractions for exact matching.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=1000,
        help="Materials Project API page size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent chemsys query workers.",
    )
    parser.add_argument(
        "--structure-batch-size",
        type=int,
        default=50,
        help="Number of material ids per structure request.",
    )
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=0,
        help="Debug limit on target formulas after aggregation. 0 means all.",
    )
    parser.add_argument(
        "--limit-chemsys",
        type=int,
        default=0,
        help="Debug limit on chemical systems after target aggregation. 0 means all.",
    )
    parser.add_argument(
        "--no-structures",
        action="store_true",
        help="Only query/filter metadata; do not download structures or write CIFs.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay before uncached API requests in each worker.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached API JSON and refetch from Materials Project.",
    )
    return parser.parse_args()


def parse_number(text: str) -> Fraction:
    if not text:
        return Fraction(1, 1)
    try:
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError) as exc:
        raise FormulaError(f"invalid numeric amount {text!r}") from exc


def add_amount(comp: OrderedDict[str, Fraction], element: str, amount: Fraction) -> None:
    if element not in comp:
        comp[element] = Fraction(0, 1)
    comp[element] += amount


def parse_formula(formula: Any) -> OrderedDict[str, Fraction]:
    if formula is None or (isinstance(formula, float) and math.isnan(formula)):
        raise FormulaError("empty formula")

    text = str(formula).strip()
    if not text:
        raise FormulaError("empty formula")

    text = (
        text.replace(" ", "")
        .replace("·", "")
        .replace("•", "")
        .replace("_", "")
    )
    stack: list[OrderedDict[str, Fraction]] = [OrderedDict()]
    bracket_stack: list[str] = []
    close_for = {"(": ")", "[": "]", "{": "}"}
    i = 0

    while i < len(text):
        ch = text[i]

        if ch in "([{":
            stack.append(OrderedDict())
            bracket_stack.append(close_for[ch])
            i += 1
            continue

        if ch in ")]}":
            if not bracket_stack or ch != bracket_stack[-1]:
                raise FormulaError(f"unmatched bracket at position {i}: {text!r}")
            group = stack.pop()
            bracket_stack.pop()
            i += 1
            start = i
            while i < len(text) and (text[i].isdigit() or text[i] == "."):
                i += 1
            factor = parse_number(text[start:i])
            for element, amount in group.items():
                add_amount(stack[-1], element, amount * factor)
            continue

        if ch.isupper():
            start = i
            i += 1
            if i < len(text) and text[i].islower():
                i += 1
            element = text[start:i]
            start_num = i
            while i < len(text) and (text[i].isdigit() or text[i] == "."):
                i += 1
            amount = parse_number(text[start_num:i])
            add_amount(stack[-1], element, amount)
            continue

        raise FormulaError(f"unexpected character {ch!r} at position {i}: {text!r}")

    if bracket_stack:
        raise FormulaError(f"unclosed bracket in formula {text!r}")

    comp = OrderedDict((el, amt) for el, amt in stack[0].items() if amt != 0)
    if not comp:
        raise FormulaError(f"no elements parsed from formula {text!r}")
    return comp


def normalized_fractions(comp: OrderedDict[str, Fraction]) -> dict[str, float]:
    total = sum(float(v) for v in comp.values())
    if total <= 0:
        raise FormulaError("non-positive composition total")
    return {el: float(amount) / total for el, amount in comp.items()}


def canonical_key(comp: OrderedDict[str, Fraction], places: int = 10) -> str:
    norm = normalized_fractions(comp)
    return "|".join(f"{el}:{norm[el]:.{places}f}" for el in sorted(norm))


def chemsys_from_comp(comp: OrderedDict[str, Fraction]) -> str:
    return "-".join(sorted(comp))


def compositions_match(
    target: OrderedDict[str, Fraction],
    candidate: OrderedDict[str, Fraction],
    tol: float,
) -> bool:
    if set(target) != set(candidate):
        return False
    target_norm = normalized_fractions(target)
    cand_norm = normalized_fractions(candidate)
    return all(abs(target_norm[el] - cand_norm[el]) <= tol for el in target_norm)


def safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    if len(text) > max_len:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        text = f"{text[: max_len - 11]}_{digest}"
    return text or "formula"


def read_targets(base_dir: Path, limit_targets: int = 0) -> tuple[list[TargetFormula], pd.DataFrame]:
    missing_path = base_dir / "missing_cif_formula_summary_after_external.csv"
    approx_path = (
        base_dir
        / "02_exact_plus_3dsc"
        / "exact_plus_3dsc_3dsc_approx_cif_inventory.csv"
    )
    if not missing_path.exists():
        raise FileNotFoundError(missing_path)
    if not approx_path.exists():
        raise FileNotFoundError(approx_path)

    raw_records: list[dict[str, Any]] = []

    missing = pd.read_csv(missing_path)
    for idx, row in missing.iterrows():
        raw_records.append(
            {
                "source_category": "missing_after_external",
                "source_record": f"missing_formula_summary:{idx + 2}",
                "formula": row.get("formula_standardized", row.get("formula_reduced")),
                "formula_reduced": row.get("formula_reduced", row.get("formula_standardized")),
                "source_rows": int(row.get("rows", 0) or 0),
                "ml_rows": int(row.get("rows", 0) or 0),
            }
        )

    approx = pd.read_csv(approx_path)
    approx = approx[approx.get("status", "generated").eq("generated")]
    for idx, row in approx.iterrows():
        raw_records.append(
            {
                "source_category": "3dsc_approx",
                "source_record": f"3dsc_approx_inventory:{idx + 2}",
                "formula": row.get("target_formula", row.get("target_formula_reduced")),
                "formula_reduced": row.get(
                    "target_formula_reduced", row.get("target_formula")
                ),
                "source_rows": int(row.get("ml_rows", 0) or 0),
                "ml_rows": int(row.get("ml_rows", 0) or 0),
            }
        )

    grouped: dict[str, dict[str, Any]] = {}
    parse_rows: list[dict[str, Any]] = []

    for rec in raw_records:
        formula = rec["formula"]
        formula_reduced = rec["formula_reduced"]
        try:
            comp = parse_formula(formula_reduced)
            key = canonical_key(comp)
            chemsys = chemsys_from_comp(comp)
            status = "parsed"
            error = ""
        except FormulaError as exc:
            comp = OrderedDict()
            key = f"parse_error:{formula_reduced}:{rec['source_record']}"
            chemsys = ""
            status = "parse_error"
            error = str(exc)

        parse_rows.append(
            {
                **rec,
                "parse_status": status,
                "parse_error": error,
                "canonical_key": key,
                "chemsys": chemsys,
            }
        )

        if status != "parsed":
            continue

        if key not in grouped:
            grouped[key] = {
                "formula": str(formula),
                "formula_reduced": str(formula_reduced),
                "composition": comp,
                "chemsys": chemsys,
                "source_categories": set(),
                "source_rows": 0,
                "ml_rows": 0,
                "source_records": [],
            }
        grouped[key]["source_categories"].add(rec["source_category"])
        grouped[key]["source_rows"] += rec["source_rows"]
        grouped[key]["ml_rows"] += rec["ml_rows"]
        grouped[key]["source_records"].append(rec["source_record"])

    sorted_items = sorted(
        grouped.items(),
        key=lambda item: (
            -item[1]["source_rows"],
            item[1]["chemsys"],
            item[1]["formula_reduced"],
        ),
    )
    if limit_targets:
        sorted_items = sorted_items[:limit_targets]

    targets: list[TargetFormula] = []
    for idx, (key, item) in enumerate(sorted_items, start=1):
        targets.append(
            TargetFormula(
                target_id=f"target_{idx:05d}",
                formula=item["formula"],
                formula_reduced=item["formula_reduced"],
                canonical_key=key,
                chemsys=item["chemsys"],
                composition=item["composition"],
                source_categories=";".join(sorted(item["source_categories"])),
                source_rows=item["source_rows"],
                ml_rows=item["ml_rows"],
                source_records=";".join(item["source_records"][:20]),
            )
        )

    parse_df = pd.DataFrame(parse_rows)
    return targets, parse_df


class MPClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        page_limit: int,
        sleep: float,
        force_refresh: bool,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.page_limit = page_limit
        self.sleep = sleep
        self.force_refresh = force_refresh
        self.session = requests.Session()
        self.session.trust_env = False
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, prefix: str, params: dict[str, Any]) -> Path:
        serialized = json.dumps(params, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{prefix}_{digest}.json"

    def get_json(self, prefix: str, params: dict[str, Any]) -> dict[str, Any]:
        path = self.cache_path(prefix, params)
        if path.exists() and not self.force_refresh:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload["_cache_status"] = "hit"
            return payload

        if self.sleep:
            time.sleep(self.sleep)

        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.session.get(
                    MP_SUMMARY_URL,
                    params=params,
                    headers={"X-API-KEY": self.api_key},
                    timeout=60,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait = min(60.0, 2.0**attempt)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                payload["_cache_status"] = "miss"
                with path.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                return payload
            except Exception as exc:  # noqa: BLE001 - preserve API failure details.
                last_exc = exc
                time.sleep(min(60.0, 2.0**attempt))

        raise RuntimeError(f"Materials Project request failed: {last_exc}")

    def query_chemsys(self, chemsys: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        docs: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        skip = 0
        total_doc: int | None = None

        while True:
            params = {
                "chemsys": chemsys,
                "_fields": SUMMARY_FIELDS,
                "_limit": self.page_limit,
                "_skip": skip,
            }
            try:
                payload = self.get_json("summary_chemsys", params)
                data = payload.get("data", [])
                meta = payload.get("meta", {})
                total_doc = meta.get("total_doc", total_doc)
                docs.extend(data)
                audit.append(
                    {
                        "query_type": "chemsys",
                        "query_value": chemsys,
                        "skip": skip,
                        "status": "ok",
                        "cache_status": payload.get("_cache_status", ""),
                        "returned": len(data),
                        "total_doc": total_doc,
                        "error": "",
                    }
                )
                if not data:
                    break
                skip += self.page_limit
                if total_doc is not None and skip >= int(total_doc):
                    break
            except Exception as exc:  # noqa: BLE001
                audit.append(
                    {
                        "query_type": "chemsys",
                        "query_value": chemsys,
                        "skip": skip,
                        "status": "error",
                        "cache_status": "",
                        "returned": 0,
                        "total_doc": total_doc,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break

        return docs, audit

    def query_structures(self, material_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not material_ids:
            return [], {
                "query_type": "structures",
                "query_value": "",
                "status": "ok",
                "returned": 0,
                "error": "",
            }
        params = {
            "material_ids": ",".join(material_ids),
            "_fields": STRUCTURE_FIELDS,
            "_limit": len(material_ids),
        }
        try:
            payload = self.get_json("summary_structures", params)
            return payload.get("data", []), {
                "query_type": "structures",
                "query_value": ",".join(material_ids),
                "status": "ok",
                "cache_status": payload.get("_cache_status", ""),
                "returned": len(payload.get("data", [])),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            return [], {
                "query_type": "structures",
                "query_value": ",".join(material_ids),
                "status": "error",
                "cache_status": "",
                "returned": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }


def query_all_chemsys(
    client: MPClient,
    chemsys_values: list[str],
    workers: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_chemsys: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []

    if workers <= 1:
        for idx, chemsys in enumerate(chemsys_values, start=1):
            docs, rows = client.query_chemsys(chemsys)
            by_chemsys[chemsys] = docs
            audit.extend(rows)
            print(f"[{idx}/{len(chemsys_values)}] {chemsys}: {len(docs)} docs")
        return by_chemsys, audit

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chemsys = {
            executor.submit(client.query_chemsys, chemsys): chemsys
            for chemsys in chemsys_values
        }
        for idx, future in enumerate(futures.as_completed(future_to_chemsys), start=1):
            chemsys = future_to_chemsys[future]
            docs, rows = future.result()
            by_chemsys[chemsys] = docs
            audit.extend(rows)
            print(f"[{idx}/{len(chemsys_values)}] {chemsys}: {len(docs)} docs")

    return by_chemsys, audit


def match_targets(
    targets: list[TargetFormula],
    docs_by_chemsys: dict[str, list[dict[str, Any]]],
    tol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    audit_rows: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    material_ids: set[str] = set()

    for target in targets:
        docs = docs_by_chemsys.get(target.chemsys, [])
        candidate_count = len(docs)
        exact_count = 0
        parse_error_count = 0

        for doc in docs:
            formula_pretty = doc.get("formula_pretty", "")
            try:
                candidate_comp = parse_formula(formula_pretty)
            except FormulaError:
                parse_error_count += 1
                continue
            if not compositions_match(target.composition, candidate_comp, tol):
                continue

            exact_count += 1
            material_id = str(doc.get("material_id", ""))
            material_ids.add(material_id)
            matches.append(
                {
                    "target_id": target.target_id,
                    "source_categories": target.source_categories,
                    "target_formula": target.formula,
                    "target_formula_reduced": target.formula_reduced,
                    "canonical_key": target.canonical_key,
                    "chemsys": target.chemsys,
                    "source_rows": target.source_rows,
                    "ml_rows": target.ml_rows,
                    "material_id": material_id,
                    "mp_formula_pretty": formula_pretty,
                    "mp_formula_anonymous": doc.get("formula_anonymous", ""),
                    "energy_above_hull": doc.get("energy_above_hull", ""),
                    "is_stable": doc.get("is_stable", ""),
                    "theoretical": doc.get("theoretical", ""),
                    "deprecated": doc.get("deprecated", ""),
                    "formation_energy_per_atom": doc.get("formation_energy_per_atom", ""),
                    "nsites": doc.get("nsites", ""),
                    "cif_path": "",
                    "download_status": "pending",
                    "error": "",
                }
            )

        status = "matched" if exact_count else "not_found"
        audit_rows.append(
            {
                "target_id": target.target_id,
                "source_categories": target.source_categories,
                "target_formula": target.formula,
                "target_formula_reduced": target.formula_reduced,
                "canonical_key": target.canonical_key,
                "chemsys": target.chemsys,
                "source_rows": target.source_rows,
                "ml_rows": target.ml_rows,
                "status": status,
                "candidate_docs_in_chemsys": candidate_count,
                "exact_mp_match_count": exact_count,
                "candidate_formula_parse_errors": parse_error_count,
                "error": "",
            }
        )

    return audit_rows, matches, material_ids


def cif_quote(text: Any) -> str:
    s = str(text).replace("'", "")
    return f"'{s}'"


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return "?"


def formula_sum_from_structure(structure: dict[str, Any]) -> str:
    counts: OrderedDict[str, float] = OrderedDict()
    for site in structure.get("sites", []):
        for species in site.get("species", []):
            element = species.get("element") or species.get("label") or species.get("symbol")
            if not element:
                continue
            occu = float(species.get("occu", 1.0))
            counts[element] = counts.get(element, 0.0) + occu
    parts = []
    for element, amount in counts.items():
        if abs(amount - round(amount)) < 1e-8:
            amount_text = str(int(round(amount)))
        else:
            amount_text = f"{amount:.8g}"
        parts.append(f"{element}{amount_text}")
    return " ".join(parts)


def structure_to_cif(
    material: dict[str, Any],
    target_formula: str,
    generated_at: str,
) -> str:
    structure = material.get("structure")
    if not isinstance(structure, dict):
        raise ValueError("missing structure dictionary")

    lattice = structure.get("lattice", {})
    sites = structure.get("sites", [])
    if not sites:
        raise ValueError("structure has no sites")

    material_id = material.get("material_id", "unknown")
    formula_pretty = material.get("formula_pretty", "")
    block_name = safe_name(str(material_id), max_len=80)
    lines = [
        f"# Generated from Materials Project on {generated_at}",
        f"# material_id: {material_id}",
        f"# mp_formula_pretty: {formula_pretty}",
        f"# target_formula: {target_formula}",
        f"data_{block_name}",
        f"_chemical_formula_sum {cif_quote(formula_sum_from_structure(structure) or formula_pretty)}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        f"_cell_length_a {format_float(lattice.get('a'))}",
        f"_cell_length_b {format_float(lattice.get('b'))}",
        f"_cell_length_c {format_float(lattice.get('c'))}",
        f"_cell_angle_alpha {format_float(lattice.get('alpha'))}",
        f"_cell_angle_beta {format_float(lattice.get('beta'))}",
        f"_cell_angle_gamma {format_float(lattice.get('gamma'))}",
        f"_cell_volume {format_float(lattice.get('volume'))}",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    label_counts: defaultdict[str, int] = defaultdict(int)
    for site in sites:
        abc = site.get("abc")
        if not abc or len(abc) != 3:
            raise ValueError("site missing fractional coordinates")
        for species in site.get("species", []):
            element = species.get("element") or species.get("label") or species.get("symbol")
            if not element:
                raise ValueError("site species missing element")
            occu = species.get("occu", 1)
            label_counts[element] += 1
            label = f"{element}{label_counts[element]}"
            lines.append(
                f"{label} {element} "
                f"{format_float(abc[0])} {format_float(abc[1])} {format_float(abc[2])} "
                f"{format_float(occu)}"
            )

    lines.append("")
    return "\n".join(lines)


def write_csv_xlsx(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    xlsx_path = csv_path.with_suffix(".xlsx")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: failed to write {xlsx_path}: {exc}", file=sys.stderr)


def download_structures_and_write_cifs(
    client: MPClient,
    matches: list[dict[str, Any]],
    output_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not matches:
        return matches, []

    generated_at = datetime.now(timezone.utc).isoformat()
    material_ids = sorted({row["material_id"] for row in matches if row["material_id"]})
    structure_docs: dict[str, dict[str, Any]] = {}
    structure_audit: list[dict[str, Any]] = []

    for start in range(0, len(material_ids), batch_size):
        batch = material_ids[start : start + batch_size]
        docs, audit = client.query_structures(batch)
        structure_audit.append(audit)
        for doc in docs:
            structure_docs[str(doc.get("material_id", ""))] = doc
        print(
            f"structures [{start + 1}-{start + len(batch)}/{len(material_ids)}]: "
            f"{len(docs)} docs"
        )

    cifs_dir = output_dir / "cifs" / "mp"
    for row in matches:
        material_id = row["material_id"]
        doc = structure_docs.get(material_id)
        if not doc:
            row["download_status"] = "structure_missing"
            row["error"] = "structure document not returned by MP"
            continue

        folder = cifs_dir / safe_name(row["target_formula_reduced"], max_len=90)
        folder.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{safe_name(row['target_formula_reduced'], max_len=90)}-MP-{safe_name(material_id)}.cif"
        )
        path = folder / filename
        try:
            cif_text = structure_to_cif(doc, row["target_formula"], generated_at)
            path.write_text(cif_text, encoding="utf-8")
            row["cif_path"] = str(path.resolve())
            row["download_status"] = "downloaded"
            row["error"] = ""
        except Exception as exc:  # noqa: BLE001
            row["download_status"] = "write_error"
            row["error"] = f"{type(exc).__name__}: {exc}"

    return matches, structure_audit


def build_summary(
    targets: list[TargetFormula],
    audit_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"metric": "target_formulas", "value": len(targets)})
    rows.append({"metric": "unique_chemsys", "value": len({target.chemsys for target in targets})})
    rows.append(
        {
            "metric": "target_formulas_missing_after_external",
            "value": sum(
                "missing_after_external" in target.source_categories for target in targets
            ),
        }
    )
    rows.append(
        {
            "metric": "target_formulas_3dsc_approx",
            "value": sum("3dsc_approx" in target.source_categories for target in targets),
        }
    )
    rows.append(
        {
            "metric": "target_formulas_with_mp_exact_match",
            "value": int((audit_df["exact_mp_match_count"] > 0).sum())
            if not audit_df.empty
            else 0,
        }
    )
    rows.append(
        {
            "metric": "target_formulas_without_mp_exact_match",
            "value": int((audit_df["exact_mp_match_count"] == 0).sum())
            if not audit_df.empty
            else 0,
        }
    )
    rows.append({"metric": "mp_exact_material_matches", "value": len(inventory_df)})
    rows.append(
        {
            "metric": "downloaded_cif_files",
            "value": int(inventory_df["download_status"].eq("downloaded").sum())
            if not inventory_df.empty and "download_status" in inventory_df
            else 0,
        }
    )
    rows.append(
        {
            "metric": "chemsys_query_errors",
            "value": int(query_df["status"].eq("error").sum())
            if not query_df.empty and "status" in query_df
            else 0,
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("error: set MP_API_KEY or pass --api-key", file=sys.stderr)
        return 2

    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    targets, parse_df = read_targets(base_dir, args.limit_targets)
    if not targets:
        print("no parseable target formulas found", file=sys.stderr)
        return 1

    chemsys_values = sorted({target.chemsys for target in targets})
    if args.limit_chemsys:
        keep = set(chemsys_values[: args.limit_chemsys])
        targets = [target for target in targets if target.chemsys in keep]
        chemsys_values = sorted(keep)

    target_df = pd.DataFrame(
        [
            {
                "target_id": target.target_id,
                "source_categories": target.source_categories,
                "target_formula": target.formula,
                "target_formula_reduced": target.formula_reduced,
                "canonical_key": target.canonical_key,
                "chemsys": target.chemsys,
                "source_rows": target.source_rows,
                "ml_rows": target.ml_rows,
                "source_records": target.source_records,
            }
            for target in targets
        ]
    )

    write_csv_xlsx(parse_df, output_dir / "mp_exact_target_parse_audit.csv")
    write_csv_xlsx(target_df, output_dir / "mp_exact_target_formulas.csv")

    print(
        f"targets: {len(targets)} formulas; chemsys: {len(chemsys_values)}; "
        f"output: {output_dir}"
    )

    client = MPClient(
        api_key=args.api_key,
        cache_dir=output_dir / "cache",
        page_limit=args.page_limit,
        sleep=args.sleep,
        force_refresh=args.force_refresh,
    )

    docs_by_chemsys, query_audit = query_all_chemsys(
        client=client,
        chemsys_values=chemsys_values,
        workers=max(1, args.workers),
    )
    query_df = pd.DataFrame(query_audit)
    write_csv_xlsx(query_df, output_dir / "mp_exact_query_audit.csv")

    audit_rows, matches, _material_ids = match_targets(
        targets=targets,
        docs_by_chemsys=docs_by_chemsys,
        tol=args.match_tol,
    )

    if args.no_structures:
        for row in matches:
            row["download_status"] = "not_requested"
    else:
        matches, structure_audit = download_structures_and_write_cifs(
            client=client,
            matches=matches,
            output_dir=output_dir,
            batch_size=args.structure_batch_size,
        )
        if structure_audit:
            structure_df = pd.DataFrame(structure_audit)
            write_csv_xlsx(structure_df, output_dir / "mp_exact_structure_query_audit.csv")

    audit_df = pd.DataFrame(audit_rows)
    inventory_df = pd.DataFrame(matches)
    write_csv_xlsx(audit_df, output_dir / "mp_exact_cif_audit.csv")
    write_csv_xlsx(inventory_df, output_dir / "mp_exact_cif_inventory.csv")

    summary_df = build_summary(targets, audit_df, inventory_df, query_df)
    write_csv_xlsx(summary_df, output_dir / "mp_exact_cif_summary.csv")

    print(summary_df.to_string(index=False))
    print(f"audit: {output_dir / 'mp_exact_cif_audit.csv'}")
    print(f"inventory: {output_dir / 'mp_exact_cif_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
