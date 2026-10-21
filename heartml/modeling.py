"""Model definitions, search spaces, cross-validation and evaluation helpers."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate

from heartml import config as C
from heartml.features import build_pipeline

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def model_specs(seed=C.RANDOM_STATE):
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=seed),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "clf__class_weight": [None, "balanced"]}),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [3, 5, 8, None],
             "clf__min_samples_leaf": [1, 3, 5]}),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.03, 0.1],
             "clf__max_depth": [2, 3]}),
    }


def tune(name, X, y, seed=C.RANDOM_STATE, cv_folds=C.CV_FOLDS, specs=None):
    estimator, grid = (specs or model_specs(seed))[name]
    search = GridSearchCV(build_pipeline(estimator), grid, scoring="roc_auc",
                          cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
                          n_jobs=1, refit=True)
    search.fit(X, y)
    return search


def cross_validate_pipeline(pipeline, X, y, seed=C.RANDOM_STATE, folds=C.CV_FOLDS, repeats=C.CV_REPEATS):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed + 1)
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    return {m: (float(np.mean(res[f"test_{m}"])), float(np.std(res[f"test_{m}"]))) for m in SCORING}


def evaluate(y_true, y_score, threshold=C.DECISION_THRESHOLD):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def pick_best(cv_table):
    """Highest CV ROC-AUC; ties broken by recall (a missed case is the costly error)."""
    return max(cv_table, key=lambda n: (round(cv_table[n]["roc_auc"][0], 4), cv_table[n]["recall"][0]))


def permutation_importances(fitted_pipeline, X, y, seed=C.RANDOM_STATE, repeats=15):
    r = permutation_importance(fitted_pipeline, X, y, scoring="roc_auc", n_repeats=repeats,
                               random_state=seed, n_jobs=1)
    return list(X.columns), r.importances_mean, r.importances_std
