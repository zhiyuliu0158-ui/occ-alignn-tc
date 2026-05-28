"""Tests for split preservation and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from occ_alignn.data.split import ensure_split


def test_existing_partial_split_is_preserved() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "parent_cif_id": ["p1", "p2", "p3", "p4"],
            "split": ["train", "train", "val", "val"],
        }
    )

    result = ensure_split(df, seed=1)

    assert result["split"].tolist() == ["train", "train", "val", "val"]


def test_invalid_existing_split_raises() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "split": ["train", "holdout"],
        }
    )

    with pytest.raises(ValueError, match="Invalid split values"):
        ensure_split(df)


def test_partially_missing_existing_split_raises() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "split": ["train", ""],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        ensure_split(df)
