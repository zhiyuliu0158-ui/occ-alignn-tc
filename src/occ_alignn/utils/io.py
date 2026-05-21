"""Small IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
