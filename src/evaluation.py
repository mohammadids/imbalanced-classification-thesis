from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer


PROBABILITY_METRICS = ("roc_auc", "average_precision")


def get_score_values(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model must expose predict_proba or decision_function for ROC-AUC and Average Precision")


def evaluate_predictions(y_true, y_pred, y_score, training_time: float, prediction_time: float) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    geometric_mean = float(np.sqrt(recall * specificity))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "geometric_mean": geometric_mean,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "training_time": training_time,
        "prediction_time": prediction_time,
    }


def predict_and_score(pipeline, X_test):
    y_pred = pipeline.predict(X_test)
    y_score = get_score_values(pipeline, X_test)
    return y_pred, y_score


def fit_predict_evaluate(pipeline, X_train, y_train, X_test, y_test) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred, y_score = predict_and_score(pipeline, X_test)
    prediction_time = time.perf_counter() - start

    return evaluate_predictions(y_test, y_pred, y_score, training_time, prediction_time), y_pred, y_score


def specificity_score(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def geometric_mean_score(y_true, y_pred) -> float:
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return float(np.sqrt(recall * specificity_score(y_true, y_pred)))


CV_SCORERS = {
    "average_precision": make_scorer(average_precision_score, response_method="predict_proba"),
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "f2": make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0),
    "mcc": make_scorer(matthews_corrcoef),
    "geometric_mean": make_scorer(geometric_mean_score),
}
