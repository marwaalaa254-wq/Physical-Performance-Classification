#!/usr/bin/env python
"""Evaluate the saved best model against the held-out test split.

Reconstructs the exact train/test split used by train.py (same seed, same
test size, same stratification) so this reports genuine held-out
performance rather than re-scoring on data the model may have trained on.

Usage:
    python evaluate.py --data data/bodyPerformance.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import models as model_defs
from src import preprocessing, utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/bodyPerformance.csv")
    p.add_argument("--model", default="models/best_model.joblib")
    p.add_argument("--scaler", default="artifacts/scaler.joblib")
    p.add_argument("--encoders", default="artifacts/label_encoders.joblib")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = utils.load_artifact(args.model)
    scaler = utils.load_artifact(args.scaler)
    encoders = utils.load_artifact(args.encoders)
    gender_encoder, class_encoder = encoders["gender"], encoders["class"]

    df = preprocessing.load_raw(args.data)
    df = preprocessing.clean(df)
    df_ml, _, _ = preprocessing.encode_labels(df, gender_encoder, class_encoder)
    X, y = preprocessing.split_features_target(df_ml)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=model_defs.RANDOM_STATE, stratify=y
    )
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Held-out test accuracy: {acc * 100:.2f}%\n")
    print(report)
    print("Confusion matrix (rows=actual, cols=predicted):", list(class_encoder.classes_))
    print(cm)

    utils.save_json(
        {"test_accuracy_pct": round(float(acc) * 100, 2)}, Path(args.results_dir) / "eval_metrics.json"
    )


if __name__ == "__main__":
    main()
