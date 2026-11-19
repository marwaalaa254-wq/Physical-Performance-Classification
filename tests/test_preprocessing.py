"""Unit tests for src.preprocessing that run without the real dataset.

Uses small synthetic DataFrames covering each cleaning rule documented in
notebooks/Body_Performance_Notebook.ipynb (Phases 5.3-5.9) so the pipeline's
behavior is pinned even though data/bodyPerformance.csv is not committed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    BEND_FORWARD_BOUNDS,
    FEATURE_COLUMNS,
    build_feature_row,
    cap_bend_forward_outliers,
    drop_impossible_rows,
    encode_labels,
    fix_dtypes,
    split_features_target,
)


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40, 60],
            "gender": ["M", "F", "M"],
            "height_cm": [175.0, 160.0, 180.0],
            "weight_kg": [70.0, 55.0, 90.0],
            "body fat_%": [15.0, 25.0, 80.0],  # row 2: impossible (>70)
            "diastolic": [80.0, 70.0, 200.0],  # row 2: impossible (>130)
            "systolic": [120.0, 110.0, 250.0],  # row 2: impossible (>200)
            "gripForce": [45.0, 30.0, 0.0],  # row 2: impossible (==0)
            "sit and bend forward_cm": [10.0, -5.0, 300.0],
            "sit-ups counts": [40, 20, 50],
            "broad jump_cm": [220.0, 150.0, 0.0],  # row 2: impossible (==0)
            "class": ["A", "C", "D"],
        }
    )


def test_fix_dtypes_casts_expected_columns():
    df = fix_dtypes(make_raw_df())
    assert str(df["gender"].dtype) == "category"
    assert str(df["age"].dtype) == "Int64"
    assert df["height_cm"].dtype.kind == "f"


def test_drop_impossible_rows_removes_bad_row():
    df = fix_dtypes(make_raw_df())
    cleaned = drop_impossible_rows(df)
    assert len(cleaned) == 2
    assert (cleaned["body fat_%"] <= 70).all()


def test_cap_bend_forward_outliers_clips_to_bounds():
    df = pd.DataFrame({"sit and bend forward_cm": [-100.0, 0.0, 300.0]})
    capped = cap_bend_forward_outliers(df)
    lower, upper = BEND_FORWARD_BOUNDS
    assert capped["sit and bend forward_cm"].min() == lower
    assert capped["sit and bend forward_cm"].max() == upper


def test_encode_labels_matches_documented_mapping():
    df = fix_dtypes(make_raw_df())
    df = drop_impossible_rows(df)
    encoded, gender_enc, _class_enc = encode_labels(df)
    mapping = dict(zip(gender_enc.classes_, gender_enc.transform(gender_enc.classes_)))
    assert mapping == {"F": 0, "M": 1}


def test_split_features_target_shapes():
    df = fix_dtypes(make_raw_df())
    df = drop_impossible_rows(df)
    encoded, _, _ = encode_labels(df)
    X, y = split_features_target(encoded)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == 2


def test_build_feature_row_accepts_string_gender():
    row = build_feature_row(
        {
            "age": 30,
            "gender": "female",
            "height_cm": 165,
            "weight_kg": 60,
            "body fat_%": 22,
            "diastolic": 75,
            "systolic": 118,
            "gripForce": 35,
            "sit and bend forward_cm": 12,
            "sit-ups counts": 30,
            "broad jump_cm": 180,
        }
    )
    assert row.loc[0, "gender"] == 0
    assert list(row.columns) == FEATURE_COLUMNS


def test_build_feature_row_raises_on_missing_feature():
    with pytest.raises(ValueError):
        build_feature_row({"age": 30})
