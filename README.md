# Physio-Intel

**Classify physical fitness from biometric measurements.**

A scikit-learn pipeline that grades an individual's physical performance
(**A** best – **D** lowest) from body measurements and fitness-test results,
built on the Body Performance Data dataset (13,393 South Korean adults).

---

## Project Overview

Physio-Intel is a 4-class classification system that predicts a performance
grade from 11 biometric and performance-test features: age, gender, height,
weight, body fat %, blood pressure (diastolic/systolic), grip force,
flexibility (sit-and-bend-forward), sit-ups count, and broad jump distance.
Five model families were benchmarked in the research notebook under
`notebooks/`: KNN, Decision Tree, SVM (RBF and linear kernels), and a
Multi-Layer Perceptron neural network. A companion Linear Regression task
predicts continuous broad-jump distance from the other features.

## Problem Statement

Gyms and fitness programs need a fast, consistent way to grade a new
member's baseline fitness level from a single measurement session, without
relying on subjective coaching assessment. A classifier trained on
biometric + performance-test data offers a repeatable first-pass grading
that can seed personalized training plans.

## Dataset

**Body Performance Data** — 13,393 rows, 12 columns (11 features + `class`
target). See [`data/README.md`](data/README.md) for the download link and
column descriptions.

After cleaning (below), 13,369 rows remain, split 80/20 (stratified,
seed=42) into:

| Split | Size |
|---|---:|
| Train | 10,695 |
| Test | 2,674 |

Class balance is near-perfect: 3,344 / 3,342 / 3,343 / 3,340 rows for
A / B / C / D respectively — no class rebalancing needed.

### Label Encoding

```
gender: F = 0, M = 1
class : A = 0, B = 1, C = 2, D = 3
```

## Data Quality Checks & Cleaning

From the notebook's EDA (Phases 5.3–5.9), applied in
[`src/preprocessing.py`](src/preprocessing.py) (`clean()`):

- No missing values, no duplicate rows.
- Dropped 24 rows with physiologically impossible readings: `body fat_% > 70`,
  `diastolic` outside `[40, 130]`, `systolic` outside `[70, 200]`,
  `gripForce == 0`, or `broad jump_cm == 0`.
- Capped `sit and bend forward_cm` to `[-3.80, 35.40]` (IQR bounds) — the raw
  data contained a physically-impossible 213 cm flexibility reading.
- Other statistical outliers (e.g. exceptionally strong grip force, very
  young/old participants) were **kept** — they represent real biological
  variation and genuine high/low performers, not data errors.

## Feature Correlations

The strongest relationships in the data (Pearson r):

| Pair | r |
|---|---:|
| `gripForce` ↔ `broad jump_cm` | 0.75 |
| `sit-ups counts` ↔ `broad jump_cm` | 0.75 |
| `weight_kg` ↔ `height_cm` | 0.74 |

`gripForce`, `sit-ups counts`, and `broad jump_cm` move together — they
capture a shared "overall physical strength" factor — while `body fat_%`
and `age` are the strongest *negative* predictors of performance class.

## Models Evaluated

| Model | Hyperparameter | Test Accuracy (80:20 split) |
|---|---|---:|
| **Neural Network (MLP)** | 1 hidden layer, 64 neurons | **74.23%** |
| SVM (RBF kernel) | `C = 5` | 71.32% |
| Decision Tree | `max_depth = 11` | 68.81% |
| SVM (linear kernel) | `C = 1` | 63.91% |
| KNN | `k = 19`, Euclidean distance | 62.75% |

These figures are the notebook's own printed output from each model's
dedicated evaluation cell, cross-checked cell-by-cell against the notebook
source — **not** the notebook's own "final comparison" summary cell, which
was found to report a fabricated 75.5% for the neural network and incorrect
hyperparameters for KNN/Decision Tree. See [`AUDIT_NOTES.md`](AUDIT_NOTES.md)
for the full discrepancy trail.

**Note on methodology:** the notebook selected each model's hyperparameter
by directly maximizing accuracy on the test set (see `AUDIT_NOTES.md` §2) —
a form of test-set leakage that optimistically biases these numbers.
`train.py` in this repo fixes that: it selects hyperparameters by 5-fold
cross-validation on the training split only, then scores the winner once on
the untouched test set. Because it wasn't run against the real dataset in
this refactor (no `data/bodyPerformance.csv` in this environment — see
[Reproducing These Results](#reproducing-these-results)), its own numbers
may differ slightly from the table above; run `python train.py` to regenerate
`results/metrics.json` with fully-verified figures.

## Best Model: Neural Network (MLP)

```
Input (11 features, StandardScaler-normalized)
    │
Dense(64, activation=relu)
    │
Dense(4, softmax)   -- solver=adam, alpha=0.001, early_stopping=True
```

Full config: [`src/models.py`](src/models.py) (`build_mlp`). Converged in 83
epochs with early stopping (`n_iter_no_change=20`, `validation_fraction=0.1`).

## Bonus: Regression — Predicting Broad Jump Distance

A separate Linear Regression model predicts `broad_jump_cm` (continuous)
from the other 10 features — useful for setting a personalized jump-distance
baseline before training begins.

| Metric | Score |
|---|---:|
| R² (test) | 0.786 |
| MAE (test) | 13.87 cm |
| RMSE (test) | 17.96 cm |

Strongest positive predictors: `gender` (male), `sit-ups counts`,
`gripForce`. Strongest negative predictors: `age`, `body fat_%`.

## Project Structure

```
Physio-Intel-ML-Classification/
├── README.md
├── AUDIT_NOTES.md
├── requirements.txt
├── train.py
├── evaluate.py
├── inference.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── models/
│   └── best_model.joblib          (generated by train.py)
├── artifacts/
│   ├── scaler.joblib               (generated by train.py)
│   ├── label_encoders.joblib       (generated by train.py)
│   └── feature_columns.json        (generated by train.py)
├── notebooks/
│   └── Body_Performance_Notebook.ipynb
├── results/
│   ├── metrics.json                (generated by train.py)
│   ├── model_comparison.json       (generated by train.py)
│   ├── classification_report.json  (generated by train.py)
│   └── confusion_matrix.json       (generated by train.py)
├── data/
│   └── README.md
└── tests/
    └── test_preprocessing.py
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Placement

Download the Body Performance Data CSV and place it at `data/bodyPerformance.csv`.
See [`data/README.md`](data/README.md) for the source link and column
descriptions.

## Training

```bash
python train.py --data data/bodyPerformance.csv
```

Sweeps each model family's hyperparameter grid via 5-fold cross-validation
on the training split, fits the winner, and saves `models/best_model.joblib`,
`artifacts/scaler.joblib`, `artifacts/label_encoders.joblib`, and metrics/
reports under `results/`.

## Evaluation

```bash
python evaluate.py --data data/bodyPerformance.csv
```

Reconstructs the exact train/test split used by `train.py` (same seed) and
reports held-out accuracy, a full classification report, and a confusion
matrix.

## Inference

```bash
python inference.py --age 28 --gender M --height-cm 175 --weight-kg 72 \
    --body-fat 18.5 --diastolic 78 --systolic 122 --grip-force 45 \
    --bend-forward 15 --situps 42 --broad-jump 210
```

```
Predicted performance class: A

Class probabilities:
  A: 87.42%
  B: 9.11%
  C: 2.61%
  D: 0.86%
```

(Illustrative output shape — exact probabilities depend on the trained
model; run the command above after `python train.py` to get real numbers.)

## Reproducing These Results

This refactor was built without downloading the dataset into this
environment, so `train.py`/`evaluate.py` have been smoke-tested against a
small synthetic CSV (proving the pipeline runs end-to-end and every script
exits cleanly) but not yet re-run against the real
`data/bodyPerformance.csv`. The results table above cites the source
notebook's own verified per-model cells (see `AUDIT_NOTES.md`). Running
`python train.py` locally with the real dataset will regenerate
`results/metrics.json` with fully independent, reproducible figures.

## Reproducibility

- Random seed: **42** (Python `random`, NumPy, and all `sklearn` splits/estimators).
- `StandardScaler` is fit on the training split only, then applied to both
  splits — no scaling leakage from test into train.
- Hyperparameters are selected by cross-validation on the training split
  only (see `AUDIT_NOTES.md` §2 for why this differs from the notebook).
- The same `src/preprocessing.py` functions are shared by `train.py`,
  `evaluate.py`, and `inference.py` to prevent train/serve skew.

## Limitations

- **Population-specific.** The dataset is drawn from South Korean adults;
  performance grading may not transfer to other populations without
  retraining.
- **Not a medical or diagnostic tool.** Blood pressure and body fat figures
  are used purely as performance-class predictors, not clinical assessments.
- **Snapshot, not longitudinal.** The model grades a single measurement
  session — it does not track improvement over time.
- **Ceiling around 74% accuracy.** Even the best model does not cleanly
  separate the 4 classes (see `results/confusion_matrix.json` after
  training) — classes B and C in particular overlap, since fitness level is
  a continuum being forced into 4 discrete bins.

## Responsible-Use Disclaimer

Physio-Intel is a research and educational project. Do not use its
predictions as the sole basis for medical, insurance, employment, or other
consequential decisions about an individual. Pair automated grading with
qualified human assessment.

## Team

Marwa, Radwa, Basma, Yomna (see `notebooks/Body_Performance_Notebook.ipynb`
and the presentation deck for the original course submission roster).

## Report & Presentation

- [`Physical_Performance_Classification_Technical_Report.pdf`](Physical_Performance_Classification_Technical_Report.pdf) — full written report.
- [`AI_Human_Performance_Analytics ML Projct 5.pptx`](<AI_Human_Performance_Analytics ML Projct 5.pptx>) — presentation deck.

## License

No license has been specified for this project. All rights reserved by the
author(s) unless a `LICENSE` file is added.
