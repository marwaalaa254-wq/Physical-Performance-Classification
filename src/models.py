"""Model definitions and hyperparameter grids for the classification and
regression tasks evaluated in notebooks/Body_Performance_Notebook.ipynb.

Grids mirror the ranges the notebook swept manually (KNN k=1..19 odd,
DecisionTree depth, SVM C, MLP architectures) so train.py reproduces the same
search space -- but selects the winner via cross-validation on the training
split only, never by looking at the held-out test set (see AUDIT_NOTES.md).
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

KNN_K_GRID = list(range(1, 21, 2))          # odd K only: avoids 4-class tie votes
DT_DEPTH_GRID = list(range(2, 21))
SVM_C_GRID = [0.1, 1, 5, 10, 50]
MLP_ARCHITECTURE_GRID = [(16,), (32,), (64,), (128,), (64, 32)]


def build_knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, metric="euclidean")


def build_decision_tree(max_depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)


def build_svm_rbf(c: float) -> SVC:
    return SVC(kernel="rbf", C=c, probability=True, random_state=RANDOM_STATE)


def build_svm_linear(c: float) -> SVC:
    return SVC(kernel="linear", C=c, probability=True, random_state=RANDOM_STATE)


def build_mlp(hidden_layer_sizes: tuple) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def build_broad_jump_regressor() -> LinearRegression:
    return LinearRegression(fit_intercept=True)


# (family name, hyperparameter grid, model builder) -- the search space train.py sweeps.
CLASSIFIER_FAMILIES = [
    ("knn", KNN_K_GRID, build_knn),
    ("decision_tree", DT_DEPTH_GRID, build_decision_tree),
    ("svm_rbf", SVM_C_GRID, build_svm_rbf),
    ("svm_linear", SVM_C_GRID, build_svm_linear),
    ("mlp", MLP_ARCHITECTURE_GRID, build_mlp),
]
