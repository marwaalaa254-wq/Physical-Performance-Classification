"""Data loading, cleaning, and encoding for the Body Performance dataset.

Mirrors the cleaning decisions made in notebooks/Body_Performance_Notebook.ipynb
(Phases 5.3-5.9), so that train.py / evaluate.py / inference.py all share one
canonical pipeline instead of re-deriving it ad hoc.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_NUMERIC_COLUMNS = [
    "height_cm",
    "weight_kg",
    "body fat_%",
    "diastolic",
    "systolic",
    "gripForce",
    "sit and bend forward_cm",
    "broad jump_cm",
]
RAW_INTEGER_COLUMNS = ["age", "sit-ups counts"]

# Order matters: this is the exact column order the shipped model expects.
FEATURE_COLUMNS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body fat_%",
    "diastolic",
    "systolic",
    "gripForce",
    "sit and bend forward_cm",
    "sit-ups counts",
    "broad jump_cm",
]
TARGET_COLUMN = "class"
CLASS_LABELS = ["A", "B", "C", "D"]

# sit_and_bend_forward_cm outlier bounds (IQR capping, see notebook 5.9).
BEND_FORWARD_BOUNDS = (-3.80, 35.40)


def load_raw(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].astype("category")
    df["class"] = df["class"].astype("category")
    for col in RAW_INTEGER_COLUMNS:
        df[col] = df[col].round().astype("Int64")
    for col in RAW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_impossible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop physiologically impossible readings (notebook 5.6)."""
    bad = (
        (df["body fat_%"] > 70)
        | (df["diastolic"] > 130)
        | (df["diastolic"] < 40)
        | (df["systolic"] > 200)
        | (df["systolic"] < 70)
        | (df["gripForce"] == 0)
        | (df["broad jump_cm"] == 0)
    )
    return df.loc[~bad].reset_index(drop=True)


def cap_bend_forward_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap sit-and-bend-forward to realistic bounds (notebook 5.9)."""
    df = df.copy()
    lower, upper = BEND_FORWARD_BOUNDS
    df["sit and bend forward_cm"] = df["sit and bend forward_cm"].clip(lower=lower, upper=upper)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dtypes -> dedup -> drop impossible -> cap outliers."""
    df = fix_dtypes(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = drop_impossible_rows(df)
    df = cap_bend_forward_outliers(df)
    return df


def encode_labels(
    df: pd.DataFrame,
    gender_encoder: LabelEncoder | None = None,
    class_encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Encode gender (F=0, M=1) and class (A=0, B=1, C=2, D=3).

    Pass fitted encoders (from training) to encode new data consistently;
    leave them None to fit fresh ones (training time only).
    """
    df = df.copy()
    if gender_encoder is None:
        gender_encoder = LabelEncoder().fit(df["gender"])
    if class_encoder is None:
        class_encoder = LabelEncoder().fit(df["class"])
    df["gender"] = gender_encoder.transform(df["gender"])
    df["class"] = class_encoder.transform(df["class"])
    return df, gender_encoder, class_encoder


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def build_feature_row(values: dict) -> pd.DataFrame:
    """Build a single-row DataFrame in FEATURE_COLUMNS order from a dict of raw values.

    `gender` may be passed as 'M'/'F' (case-insensitive) or already-encoded 0/1.
    """
    row = dict(values)
    gender = row.get("gender")
    if isinstance(gender, str):
        row["gender"] = 1 if gender.strip().upper().startswith("M") else 0
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    return pd.DataFrame([{c: row[c] for c in FEATURE_COLUMNS}])
