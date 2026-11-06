import numpy as np

from src.evaluation import evaluate_predictions, get_score_values
from src.experiments import build_pipeline, cross_validate_on_training_data, make_train_test_split


class ProbabilityModel:
    def predict_proba(self, X):
        self.used_predict_proba = True
        return np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])


def test_confusion_matrix_values_are_mapped_correctly():
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_score=[0.1, 0.7, 0.4, 0.8],
        training_time=0.1,
        prediction_time=0.01,
    )
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1


def test_reported_metrics_are_between_zero_and_one_where_applicable():
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
        training_time=0.1,
        prediction_time=0.01,
    )
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "f2", "roc_auc", "average_precision", "geometric_mean"):
        assert 0.0 <= metrics[key] <= 1.0


def test_probability_metrics_use_probability_scores():
    model = ProbabilityModel()
    scores = get_score_values(model, X=np.zeros((4, 2)))
    assert np.allclose(scores, [0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 0, 0],
        y_score=scores,
        training_time=0.1,
        prediction_time=0.01,
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_cross_validation_outputs_means_and_standard_deviations(bundles):
    bundle = bundles["satimage"]
    X_train, _, y_train, _ = make_train_test_split(bundle.X, bundle.y)
    pipeline = build_pipeline("logistic_regression", "original", y_train=y_train)
    scores = cross_validate_on_training_data(pipeline, X_train, y_train, n_splits=2)
    for metric in ("average_precision", "roc_auc", "balanced_accuracy", "precision", "recall", "specificity", "f1", "f2", "mcc", "geometric_mean"):
        assert f"cv_{metric}_mean" in scores
        assert f"cv_{metric}_std" in scores


def test_no_legacy_average_precision_alias_in_metric_schema():
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
        training_time=0.1,
        prediction_time=0.01,
    )
    legacy_name = "pr" + "_auc"
    assert "average_precision" in metrics
    assert legacy_name not in metrics
