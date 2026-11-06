from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE


def calculate_scale_pos_weight(y_train) -> float:
    y = np.asarray(y_train)
    negatives = np.sum(y == 0)
    positives = np.sum(y == 1)
    if positives == 0:
        raise ValueError("Cannot calculate scale_pos_weight with zero positive samples")
    return float(negatives / positives)


class FoldAwareXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper that calculates scale_pos_weight inside fit()."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        use_scale_pos_weight=False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity
        self.use_scale_pos_weight = use_scale_pos_weight

    def fit(self, X, y):
        from xgboost import XGBClassifier

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
        }
        if self.use_scale_pos_weight:
            self.scale_pos_weight_ = calculate_scale_pos_weight(y)
            params["scale_pos_weight"] = self.scale_pos_weight_
        else:
            self.scale_pos_weight_ = None
        self.model_ = XGBClassifier(**params)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def get_model(classifier_name: str, treatment: str = "original", y_train=None):
    class_weight = "balanced" if treatment == "class_weighting" else None

    if classifier_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )

    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if classifier_name == "xgboost":
        return FoldAwareXGBClassifier(use_scale_pos_weight=treatment == "class_weighting")

    raise ValueError(f"Unsupported classifier {classifier_name!r}")
