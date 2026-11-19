#!/usr/bin/env python
"""Predict a performance class (A-D) for one participant's measurements.

Loads the saved best_model.joblib, scaler, and label encoders (produced by
train.py) and runs them on a single set of raw measurements -- the SAME
preprocessing/scaling path used during training, via src.preprocessing.

Usage:
    python inference.py --age 28 --gender M --height-cm 175 --weight-kg 72 \\
        --body-fat 18.5 --diastolic 78 --systolic 122 --grip-force 45 \\
        --bend-forward 15 --situps 42 --broad-jump 210
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import preprocessing, utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/best_model.joblib")
    p.add_argument("--scaler", default="artifacts/scaler.joblib")
    p.add_argument("--encoders", default="artifacts/label_encoders.joblib")
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--gender", required=True, help="M or F")
    p.add_argument("--height-cm", type=float, required=True, dest="height_cm")
    p.add_argument("--weight-kg", type=float, required=True, dest="weight_kg")
    p.add_argument("--body-fat", type=float, required=True, dest="body_fat_pct")
    p.add_argument("--diastolic", type=float, required=True)
    p.add_argument("--systolic", type=float, required=True)
    p.add_argument("--grip-force", type=float, required=True, dest="grip_force")
    p.add_argument("--bend-forward", type=float, required=True, dest="bend_forward_cm")
    p.add_argument("--situps", type=int, required=True, dest="situps_count")
    p.add_argument("--broad-jump", type=float, required=True, dest="broad_jump_cm")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = utils.load_artifact(args.model)
    scaler = utils.load_artifact(args.scaler)
    encoders = utils.load_artifact(args.encoders)
    class_encoder = encoders["class"]

    row = preprocessing.build_feature_row(
        {
            "age": args.age,
            "gender": args.gender,
            "height_cm": args.height_cm,
            "weight_kg": args.weight_kg,
            "body fat_%": args.body_fat_pct,
            "diastolic": args.diastolic,
            "systolic": args.systolic,
            "gripForce": args.grip_force,
            "sit and bend forward_cm": args.bend_forward_cm,
            "sit-ups counts": args.situps_count,
            "broad jump_cm": args.broad_jump_cm,
        }
    )
    row_scaled = scaler.transform(row)

    pred = model.predict(row_scaled)[0]
    label = class_encoder.inverse_transform([pred])[0]
    print(f"Predicted performance class: {label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        print("\nClass probabilities:")
        for cls, p_ in sorted(zip(class_encoder.classes_, proba), key=lambda kv: -kv[1]):
            print(f"  {cls}: {p_ * 100:.2f}%")


if __name__ == "__main__":
    main()
