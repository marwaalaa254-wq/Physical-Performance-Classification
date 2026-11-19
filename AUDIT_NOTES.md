# Audit Notes

This document records discrepancies found while turning
`notebooks/Body_Performance_Notebook.ipynb` into the production pipeline
(`src/`, `train.py`, `evaluate.py`, `inference.py`). It exists so the
numbers in `README.md` can be traced back to their source, and so nobody
re-discovers the same bugs later.

## 1. The notebook's own "final comparison" cell reports a fabricated number

Cell "Compare all 3 models including the neural network" (the one that
prints *"🏆 Best Model Results: Neural Network ... 75.5%"*) does **not**
read its numbers from the models actually trained earlier in the notebook.
It hardcodes them in a comment block:

```python
# Define the accuracies for 80:20 split based on your results
knn_accuracy = 62.75 / 100
dt_accuracy  = 68.81 / 100
svm_accuracy = 71.32 / 100
nn_accuracy  = 75.5 / 100    # 75.5% for layer(1):64

best_k     = 5    # Assuming k=5 for KNN
best_depth = 10   # Assuming depth=10 for Decision Tree
```

Comparing this against the cells that actually trained and scored each
model on the 80:20 split:

| Value | Hardcoded in "final comparison" cell | Actually produced earlier in the notebook |
|---|---|---|
| KNN best K | `5` | `19` (tuning cell explicitly prints `Best K: 19`) |
| Decision Tree best depth | `10` | `11` (`max_depth = 11` printed by the DT cell) |
| Neural Network test accuracy | `75.5%` | `74.23%` (the real `nn_accuracy` printed right after fitting the `(64,)` MLP) |

KNN, Decision Tree, and SVM RBF's hardcoded numbers happen to match their
real cells (SVM's `best_C=5` is correct); the Neural Network figure and both
KNN/DT "best" hyperparameters do not. Because this hardcoded cell is also
the one that renders the notebook's headline bar chart and "best model"
banner, **75.5% is the number a reader walks away with**, even though no
model in the notebook ever actually scored that.

**What this repo does instead:** `results/metrics.json` (produced by
`train.py`) is generated directly from `model.predict()` on the held-out
test split — there is no hand-typed accuracy anywhere in the pipeline.
README.md's results table cites the *verified* 74.23% figure from the real
`(64,)` MLP cell, not the notebook's own summary banner.

## 2. Hyperparameters were selected by reading the test set, not a validation set

Every tuning loop in the notebook (KNN's `k`, Decision Tree's `max_depth`,
SVM's `C`, the MLP's hidden-layer size) picks the value that maximizes
accuracy computed **directly on `X_test`/`y_test`**:

```python
for k in k_values:
    knn.fit(X_train_scaled, y_train)
    test_scores.append(knn.score(X_test_scaled, y_test))   # <- test set
best_k = max(odd_k_scores, key=odd_k_scores.get)
```

This is model selection using the test set, which optimistically biases the
reported test accuracy — the "held-out" score has effectively been used
twice (once to pick the winning hyperparameter, once to report performance).

**What this repo does instead:** `train.py` selects every hyperparameter by
5-fold cross-validation **on the training split only** (`select_best_hyperparameter`
in `train.py`), then fits the winning configuration once and scores it a
single time on the untouched test split. The reported test accuracy in
`results/metrics.json` has not been used for any selection decision.

## 3. The K-Fold comparison table mixes models from different splits

Cell "K-Fold cross validation for all models" evaluates whichever
`knn_model` / `dt_model` / `svm_model` / `nn_model` objects happen to still
be bound in memory. Because the notebook re-runs the KNN/DT/SVM/MLP sections
three times (once per 80:20, 70:30, 50:50 split) reusing the same variable
names, by the time this cell runs those variables hold the **50:50 split's**
models (`svm_lin` at `C=0.1`, MLP at `(32,)`) — not the 80:20 configuration
the table is presented next to. The cross-validation scores in that table
therefore don't correspond to any of the headline 80:20 numbers reported
elsewhere in the notebook.

**What this repo does instead:** each model family in `train.py` is tuned,
fit, and scored inside one loop iteration with no shared mutable state
across splits, so there's no possibility of silently scoring a stale model.

## What's unaffected

The EDA (Phases 5.1–5.9), the cleaning decisions (drop rows with impossible
blood pressure / zero grip force / zero broad jump, cap
`sit and bend forward_cm` to `[-3.80, 35.40]`), the label encodings
(`gender`: F=0, M=1; `class`: A=0, B=1, C=2, D=3), and the bonus Linear
Regression task (`broad_jump_cm` prediction, R²=0.786) all reproduce cleanly
against their source cells and are carried into `src/preprocessing.py`
and `src/models.py` unchanged.
