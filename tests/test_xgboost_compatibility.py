import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils import get_tags

from src.data_loader import DatasetBundle
from src.evaluation import CV_SCORERS, get_score_values
from src.experiments import build_pipeline, run_single_experiment
from src.models import FoldAwareXGBClassifier


def make_synthetic_bundle():
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.85, 0.15],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{idx:02d}" for idx in range(X.shape[1])])
    y = pd.Series(y, name="target")
    return DatasetBundle(
        name="synthetic",
        fetch_key="synthetic",
        X=X,
        y=y,
        original_y=y.copy(),
        metadata={"modeling_ready": True},
        raw_X=X.copy(),
        raw_original_y=y.copy(),
        raw_metadata={},
        cleaning_metadata={"cleaning_status": "unchanged"},
    )


def test_fold_aware_xgboost_is_sklearn_classifier_with_classifier_tags():
    model = FoldAwareXGBClassifier(n_estimators=2)
    assert is_classifier(model)
    assert get_tags(model).estimator_type == "classifier"


def test_fold_aware_xgboost_pipeline_is_sklearn_classifier():
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", FoldAwareXGBClassifier(n_estimators=2)),
    ])
    assert is_classifier(pipeline)
    assert get_tags(pipeline).estimator_type == "classifier"


def test_fold_aware_xgboost_clone_fit_probability_and_class_weight_tags():
    cloned = clone(FoldAwareXGBClassifier(n_estimators=2, use_scale_pos_weight=True))
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=42,
    )
    cloned.fit(X, y)
    probabilities = cloned.predict_proba(X[:5])
    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert list(cloned.classes_) == [0, 1]
    assert cloned.scale_pos_weight_ > 1.0

    original = FoldAwareXGBClassifier(n_estimators=2, use_scale_pos_weight=False).fit(X, y)
    assert original.scale_pos_weight_ is None


def test_xgboost_cross_validate_runs_all_project_probability_scorers():
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.85, 0.15],
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    expected_metrics = {
        "average_precision",
        "roc_auc",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f2",
        "mcc",
        "geometric_mean",
    }

    for treatment in ("original", "class_weighting"):
        pipeline = build_pipeline("xgboost", treatment, y_train=y)
        assert is_classifier(pipeline)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=CV_SCORERS,
            error_score="raise",
        )
        for metric in expected_metrics:
            values = scores[f"test_{metric}"]
            assert values.shape == (2,)
            assert np.isfinite(values).all()


def test_xgboost_probability_metrics_use_positive_class_probabilities():
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=42,
    )
    pipeline = build_pipeline("xgboost", "original", y_train=y)
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X[:8])
    assert np.allclose(get_score_values(pipeline, X[:8]), probabilities[:, 1])


def test_xgboost_original_and_class_weighting_configurations_complete():
    bundle = make_synthetic_bundle()
    for treatment in ("original", "class_weighting"):
        row, cv_row, predictions = run_single_experiment(bundle, "xgboost", treatment, cv_splits=2)
        assert row["execution_status"] == "success"
        assert cv_row["execution_status"] == "success"
        assert predictions is not None
        assert "average_precision" in row
        assert "cv_average_precision_mean" in cv_row
