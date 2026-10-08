"""
models/classifiers.py
Baseline classifiers used across experiment arms:
  - XGBoost     (main classifier — handles tabular features well)
  - SVM-RBF     (strong baseline for radiomics)
  - MLP         (neural baseline, matches fusion MLP depth)

All wrapped to output predict_proba for consistent AUC computation.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed. Run: pip install xgboost")

import sys
sys.path.append("..")
import config


def get_xgboost(scale_pos_weight: float = 1.0) -> "XGBClassifier":
    """
    XGBoost with sensible defaults for tabular medical data.
    scale_pos_weight handles class imbalance: n_neg / n_pos
    """
    assert XGB_AVAILABLE, "Install xgboost: pip install xgboost"
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=config.SEED,
        n_jobs=-1,
    )


def get_svm() -> Pipeline:
    """
    SVM with RBF kernel in a scaling pipeline.
    probability=True enables predict_proba.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=config.SEED,
            class_weight="balanced",
        ))
    ])


def get_mlp() -> Pipeline:
    """
    Sklearn MLP baseline (shallower than the fusion MLP).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            dropout=0.3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=config.SEED,
        ))
    ])


def get_xgboost_tuned(X_train: np.ndarray,
                      y_train: np.ndarray) -> "XGBClassifier":
    """
    XGBoost with lightweight hyperparameter grid search (3-fold CV).
    Use this for final reported results.
    """
    assert XGB_AVAILABLE
    param_grid = {
        "max_depth":       [3, 4, 6],
        "learning_rate":   [0.01, 0.05, 0.1],
        "n_estimators":    [200, 300],
        "subsample":       [0.7, 0.9],
    }
    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="auc",
        random_state=config.SEED,
        n_jobs=-1,
    )
    gs = GridSearchCV(
        base, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"  Best XGB params: {gs.best_params_}")
    print(f"  Best CV AUC:     {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─── Classifier Registry ─────────────────────────────────────────────────────

CLASSIFIERS = {
    "xgboost": get_xgboost,
    "svm":     get_svm,
    "mlp":     get_mlp,
}

def get_classifier(name: str = "xgboost", **kwargs):
    assert name in CLASSIFIERS, f"Unknown classifier: {name}. Choose from {list(CLASSIFIERS)}"
    return CLASSIFIERS[name](**kwargs)
