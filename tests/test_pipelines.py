import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments import build_pipeline, make_train_test_split
from src.samplers import AdaptiveSMOTE


def test_train_test_split_is_stratified(bundles):
    for bundle in bundles.values():
        if not bundle.metadata["modeling_ready"]:
            continue
        _, _, y_train, y_test = make_train_test_split(bundle.X, bundle.y)
        full_ratio = float(np.mean(bundle.y))
        train_ratio = float(np.mean(y_train))
        test_ratio = float(np.mean(y_test))
        assert abs(full_ratio - train_ratio) < 0.01
        assert abs(full_ratio - test_ratio) < 0.01


def test_original_and_class_weighting_have_no_sampler(bundles):
    y_train = next(iter(bundles.values())).y
    for treatment in ("original", "class_weighting"):
        pipeline = build_pipeline("logistic_regression", treatment, y_train=y_train)
        assert isinstance(pipeline, Pipeline)
        assert "sampler" not in pipeline.named_steps


def test_smote_is_inside_imbalanced_learn_pipeline(bundles):
    y_train = next(iter(bundles.values())).y
    pipeline = build_pipeline("logistic_regression", "smote", y_train=y_train)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["sampler"], AdaptiveSMOTE)


def test_samplers_are_not_applied_to_test_set(bundles):
    bundle = bundles["satimage"]
    X_train, X_test, y_train, y_test = make_train_test_split(bundle.X, bundle.y)
    original_test_counts = y_test.value_counts().sort_index().to_dict()
    pipeline = build_pipeline("logistic_regression", "random_oversampling", y_train=y_train)
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)
    assert y_test.value_counts().sort_index().to_dict() == original_test_counts
    assert X_test.shape[0] == y_test.shape[0]


def test_preprocessing_is_fitted_only_on_training_data(bundles):
    bundle = bundles["satimage"]
    X_train, X_test, y_train, _ = make_train_test_split(bundle.X, bundle.y)
    pipeline = build_pipeline("logistic_regression", "original", y_train=y_train)
    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps["preprocess"]
    assert np.allclose(scaler.mean_, X_train.mean(axis=0).to_numpy())
    assert not np.allclose(scaler.mean_, bundle.X.mean(axis=0).to_numpy())
    assert X_test.shape[0] > 0


def test_identical_feature_vectors_are_not_split_between_train_and_test(bundles):
    for bundle in bundles.values():
        if not bundle.metadata["modeling_ready"]:
            continue
        X_train, X_test, _, _ = make_train_test_split(bundle.X, bundle.y)
        train_rows = set(map(tuple, X_train.to_numpy()))
        test_rows = set(map(tuple, X_test.to_numpy()))
        assert train_rows.isdisjoint(test_rows)


def test_smote_pipeline_order_scales_before_sampler_for_all_classifiers(bundles):
    y_train = next(iter(bundles.values())).y
    for classifier in ("logistic_regression", "random_forest", "xgboost"):
        pipeline = build_pipeline(classifier, "smote", y_train=y_train)
        assert list(pipeline.named_steps) == ["preprocess", "sampler", "model"]
        assert isinstance(pipeline.named_steps["preprocess"], StandardScaler)
        assert isinstance(pipeline.named_steps["sampler"], AdaptiveSMOTE)
