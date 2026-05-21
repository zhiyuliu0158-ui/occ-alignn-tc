"""Dataset for Tc regression from occupancy-aware CIF graphs."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from occ_alignn.data.graph import GraphData, build_graph_from_cif
from occ_alignn.featurizers.occupancy import parse_conditions


def load_dataframe(data_csv: str | Path) -> pd.DataFrame:
    """Load a CSV and fill minimally required columns."""
    df = pd.read_csv(data_csv)
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"sample_{idx}" for idx in range(len(df))]
    if "cif_path" not in df.columns:
        raise ValueError("CSV must contain a cif_path column.")
    return df


def _resolve_path(path_value: object, csv_dir: Path) -> str:
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    candidate = csv_dir / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def _parse_tc(value: object, require_target: bool) -> tuple[float | None, float | None]:
    if value is None or pd.isna(value):
        if require_target:
            raise ValueError("Tc_K is required for training/evaluation.")
        return None, None
    try:
        tc_k = float(value)
    except Exception as exc:
        if require_target:
            raise ValueError(f"Invalid Tc_K value: {value!r}") from exc
        return None, None
    if math.isnan(tc_k):
        if require_target:
            raise ValueError("Tc_K is NaN.")
        return None, None
    return float(math.log1p(max(tc_k, 0.0))), tc_k


class CifTcDataset(Dataset[GraphData]):
    """Torch Dataset that converts rows into occupancy-aware crystal graphs."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: dict[str, Any],
        csv_dir: str | Path = ".",
        split: str | None = None,
        split_column: str = "split",
        require_target: bool = True,
        cache_graphs: bool = True,
    ) -> None:
        self.config = config
        self.csv_dir = Path(csv_dir)
        self.require_target = require_target
        if split is not None and split_column in dataframe.columns:
            dataframe = dataframe[dataframe[split_column].astype(str).str.lower() == split]
        self.dataframe = dataframe.reset_index(drop=True)
        self.cache_graphs = cache_graphs
        self._cache: dict[int, GraphData] = {}
        if len(self.dataframe) == 0:
            warnings.warn(f"Dataset split {split!r} is empty.", RuntimeWarning)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _build_graph(self, row: pd.Series) -> GraphData:
        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})
        cif_path = _resolve_path(row.get("cif_path"), self.csv_dir)
        graph = build_graph_from_cif(
            cif_path=cif_path,
            max_species_per_site=int(model_cfg.get("max_species_per_site", 4)),
            frac_coord_tol=float(data_cfg.get("frac_coord_tol", 1e-4)),
            cutoff_radius=float(model_cfg.get("cutoff_radius", 8.0)),
            max_neighbors=int(model_cfg.get("max_neighbors", 12)),
            n_edge_rbf=int(model_cfg.get("n_edge_rbf", 80)),
            n_angle_rbf=int(model_cfg.get("n_angle_rbf", 40)),
            include_self_edges=bool(model_cfg.get("include_self_edges", False)),
        )
        conditions = parse_conditions(row.to_dict())
        target, tc_k = _parse_tc(row.get("Tc_K"), self.require_target)
        graph.pressure = torch.as_tensor(conditions.pressure, dtype=torch.float32)
        graph.field = torch.as_tensor(conditions.field, dtype=torch.float32)
        graph.field_direction_id = conditions.field_direction_id
        graph.structure_source_id = conditions.structure_source_id
        graph.match_type_id = conditions.match_type_id
        graph.fidelity_id = conditions.fidelity_id
        graph.target = target
        graph.tc_k = tc_k
        graph.sample_id = str(row.get("sample_id", ""))
        graph.cif_path = cif_path
        graph.metadata = row.to_dict()
        graph.metadata["pressure_is_unknown"] = conditions.pressure_is_unknown
        graph.metadata["field_is_unknown"] = conditions.field_is_unknown
        return graph

    def __getitem__(self, index: int) -> GraphData:
        if self.cache_graphs and index in self._cache:
            return self._cache[index]
        graph = self._build_graph(self.dataframe.iloc[index])
        if self.cache_graphs:
            self._cache[index] = graph
        return graph
