#!/usr/bin/env python
"""Train and select the best Body Performance classifier.

Pipeline (matches notebooks/Body_Performance_Notebook.ipynb Phases 5-8, with
one methodology fix -- see AUDIT_NOTES.md):

    1. Load bodyPerformance.csv, apply the notebook's cleaning decisions
       (src.preprocessing.clean): dtype fixes, drop impossible readings,
       cap sit-and-bend-forward outliers.
    2. Encode gender/class (LabelEncoder) and split into an 80/20
       train/test set (seed=42, stratified).
    3. Fit StandardScaler on the TRAIN split only.
    4. For each model family (KNN, Decision Tree, SVM-RBF, SVM-Linear, MLP),
       sweep the notebook's own hyperparameter grid, selecting the winner by
       5-fold cross-validation accuracy on the TRAIN split only -- never by
       peeking at the held-out test set, which is the bug documented in
       AUDIT_NOTES.md.
    5. Refit each family's best config on the full train split, score once
       on the untouched test split, and save the overall best model plus
       the scaler/encoders/metrics needed to reproduce inference.

Usage:
    python train.py --data data/bodyPerformance.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import models, preprocessing, utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/bodyPerformance.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def select_best_hyperparameter(model_builder, grid, X_train, y_train, cv, seed):
    """Pick the grid value with the highest mean CV accuracy, TRAIN split only."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    best_value, best_score = None, -np.inf
    for value in grid:
        scores = cross_val_score(model_builder(value), X_train, y_train, cv=skf)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_value, best_score = value, mean_score
    return best_value, best_score


def main() -> None:
    args = parse_args()
    utils.set_seed(models.RANDOM_STATE)

    print(f"Loading {args.data} ...")
    df = preprocessing.load_raw(args.data)
    rows_before = len(df)
    df = preprocessing.clean(df)
    print(f"Cleaned dataset: {rows_before} -> {len(df)} rows")

    df_ml, gender_encoder, class_encoder = preprocessing.encode_labels(df)
    X, y = preprocessing.split_features_target(df_ml)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=models.RANDOM_STATE, stratify=y
    )
    scaler = preprocessing.fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train/test split: {len(X_train)}/{len(X_test)} (seed={models.RANDOM_STATE})")

    comparison = []
    fitted = {}
    for name, grid, builder in models.CLASSIFIER_FAMILIES:
        print(f"\nTuning {name} over {len(grid)} candidate(s) via {args.cv}-fold CV (train split only)...")
        best_value, cv_score = select_best_hyperparameter(
            builder, grid, X_train_scaled, y_train, args.cv, models.RANDOM_STATE
        )
        model = builder(best_value)
        model.fit(X_train_scaled, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        fitted[name] = model
        comparison.append(
            {
                "model": name,
                "best_hyperparameter": list(best_value) if isinstance(best_value, tuple) else best_value,
                "cv_accuracy_pct": round(float(cv_score) * 100, 2),
                "test_accuracy_pct": round(float(test_acc) * 100, 2),
            }
        )
        print(f"  best={best_value}  cv_acc={cv_score * 100:.2f}%  test_acc={test_acc * 100:.2f}%")

    comparison.sort(key=lambda r: r["cv_accuracy_pct"], reverse=True)
    best_name = comparison[0]["model"]
    best_model = fitted[best_name]
    print(
        f"\nSelected best model by CV accuracy: {best_name} "
        f"({comparison[0]['cv_accuracy_pct']}% CV, {comparison[0]['test_accuracy_pct']}% held-out test)"
    )

    y_pred_best = best_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred_best, target_names=class_encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_best).tolist()

    models_dir, artifacts_dir, results_dir = Path(args.models_dir), Path(args.artifacts_dir), Path(args.results_dir)
    utils.save_artifact(best_model, models_dir / "best_model.joblib")
    utils.save_artifact(scaler, artifacts_dir / "scaler.joblib")
    utils.save_artifact({"gender": gender_encoder, "class": class_encoder}, artifacts_dir / "label_encoders.joblib")
    utils.save_json(preprocessing.FEATURE_COLUMNS, artifacts_dir / "feature_columns.json")

    utils.save_json(comparison, results_dir / "model_comparison.json")
    utils.save_json(
        {
            "best_model": best_name,
            "cv_accuracy_pct": comparison[0]["cv_accuracy_pct"],
            "test_accuracy_pct": comparison[0]["test_accuracy_pct"],
            "random_state": models.RANDOM_STATE,
            "test_size": args.test_size,
        },
        results_dir / "metrics.json",
    )
    utils.save_json(report, results_dir / "classification_report.json")
    utils.save_json({"labels": list(class_encoder.classes_), "matrix": cm}, results_dir / "confusion_matrix.json")

    print(f"\nSaved best model      -> {models_dir / 'best_model.joblib'}")
    print(f"Saved scaler+encoders -> {artifacts_dir}/")
    print(f"Saved metrics/reports -> {results_dir}/")


if __name__ == "__main__":
    main()
