import pandas as pd
import pytest

from src.config import CLASSIFIERS, REQUESTED_DATASETS, TREATMENTS
from src.experiments import (
    artifacts_are_complete_and_valid,
    build_full_experiment_grid,
    create_experiment_id,
    experiment_is_complete,
    make_manifest_row,
    run_configurations,
    save_manifest_row,
    save_result_row,
    write_summary_tables,
)
from src.models import FoldAwareXGBClassifier


def test_full_experiment_definition_has_45_unique_configurations():
    grid = build_full_experiment_grid(REQUESTED_DATASETS.keys())
    assert len(grid) == 45
    assert grid["experiment_id"].is_unique
    assert set(grid["classifier"]) == set(CLASSIFIERS)
    assert set(grid["treatment"]) == set(TREATMENTS)


def test_duplicate_experiment_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate experiment IDs"):
        build_full_experiment_grid(["satimage", "satimage"], classifiers=("logistic_regression",), treatments=("original",))


def test_completed_results_are_not_silently_overwritten(tmp_path):
    output = tmp_path / "results.csv"
    row = {
        "experiment_id": create_experiment_id("satimage", "logistic_regression", "original"),
        "dataset": "satimage",
        "classifier": "logistic_regression",
        "treatment": "original",
        "execution_status": "success",
    }
    save_result_row(row, output)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_result_row(row, output)
    saved = pd.read_csv(output)
    assert len(saved) == 1


def test_failed_configurations_may_be_retried(tmp_path):
    output = tmp_path / "results.csv"
    row = {
        "experiment_id": "demo",
        "dataset": "satimage",
        "classifier": "logistic_regression",
        "treatment": "original",
        "execution_status": "failed",
    }
    save_result_row(row, output)
    row["execution_status"] = "success"
    save_result_row(row, output)
    saved = pd.read_csv(output)
    assert len(saved) == 1
    assert saved.loc[0, "execution_status"] == "success"


def test_successful_configurations_are_skipped(tmp_path, bundles):
    output = tmp_path / "results.csv"
    experiment_id = create_experiment_id("satimage", "logistic_regression", "original")
    save_result_row({
        "experiment_id": experiment_id,
        "dataset": "satimage",
        "classifier": "logistic_regression",
        "treatment": "original",
        "execution_status": "success",
    }, output)
    grid = build_full_experiment_grid(["satimage"], classifiers=("logistic_regression",), treatments=("original",))
    result = run_configurations({"satimage": bundles["satimage"]}, grid, output, cv_splits=0, save_predictions=False)
    assert len(result) == 1


def test_atomic_write_preserves_previous_results_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "results.csv"
    first = {
        "experiment_id": "demo",
        "dataset": "satimage",
        "classifier": "logistic_regression",
        "treatment": "original",
        "execution_status": "failed",
    }
    save_result_row(first, output)

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("src.experiments.os.replace", boom)
    second = dict(first)
    second["execution_status"] = "success"
    with pytest.raises(OSError, match="simulated replace failure"):
        save_result_row(second, output)

    saved = pd.read_csv(output)
    assert saved.loc[0, "execution_status"] == "failed"


def test_xgboost_scale_pos_weight_is_calculated_inside_fit():
    model_a = FoldAwareXGBClassifier(n_estimators=1, use_scale_pos_weight=True)
    model_b = FoldAwareXGBClassifier(n_estimators=1, use_scale_pos_weight=True)
    X = [[0], [1], [2], [3], [4], [5]]
    model_a.fit(X, [0, 0, 0, 1, 1, 1])
    model_b.fit(X, [0, 0, 0, 0, 0, 1])
    assert model_a.scale_pos_weight_ == 1.0
    assert model_b.scale_pos_weight_ == 5.0


def test_manifest_completion_requires_all_artifacts(tmp_path, bundles):
    bundle = bundles["satimage"]
    experiment_id = create_experiment_id("satimage", "logistic_regression", "original")
    config = {
        "experiment_id": experiment_id,
        "dataset": "satimage",
        "classifier": "logistic_regression",
        "treatment": "original",
    }
    manifest = tmp_path / "experiment_manifest.csv"
    test_path = tmp_path / "test_results.csv"
    cv_path = tmp_path / "cross_validation_results.csv"
    predictions_dir = tmp_path / "predictions"

    save_manifest_row(make_manifest_row(config, "complete", "start", completed_at="end", artifacts_validated=True), manifest)
    assert not experiment_is_complete(bundle, config, manifest, test_path, cv_path, predictions_dir)


def test_interrupted_configurations_are_rerun_until_manifest_complete(tmp_path, bundles):
    grid = build_full_experiment_grid(["satimage"], classifiers=("logistic_regression",), treatments=("original",))
    output = tmp_path / "test_results.csv"
    cv_output = tmp_path / "cross_validation_results.csv"
    manifest = tmp_path / "experiment_manifest.csv"
    predictions_dir = tmp_path / "predictions"

    result = run_configurations(
        {"satimage": bundles["satimage"]},
        grid,
        output,
        cv_output_path=cv_output,
        manifest_path=manifest,
        prediction_output_dir=predictions_dir,
        save_predictions=True,
        cv_splits=2,
    )
    assert result.loc[0, "execution_status"] == "success"
    experiment_id = result.loc[0, "experiment_id"]
    config = grid.iloc[0].to_dict()
    assert experiment_is_complete(bundles["satimage"], config, manifest, output, cv_output, predictions_dir)

    save_manifest_row(make_manifest_row(config, "running", "restart"), manifest)
    rerun = run_configurations(
        {"satimage": bundles["satimage"]},
        grid,
        output,
        cv_output_path=cv_output,
        manifest_path=manifest,
        prediction_output_dir=predictions_dir,
        save_predictions=True,
        cv_splits=2,
    )
    assert rerun.loc[rerun["experiment_id"] == experiment_id, "execution_status"].iloc[0] == "success"
    assert experiment_is_complete(bundles["satimage"], config, manifest, output, cv_output, predictions_dir)


def test_ranking_uses_within_dataset_rank_before_cross_dataset_average(tmp_path):
    cv = pd.DataFrame([
        {"experiment_id": "d1__a__x", "dataset": "d1", "classifier": "a", "treatment": "x", "positive_label": 1, "execution_status": "success", "cv_average_precision_mean": 0.9, "cv_f1_mean": 0.3, "cv_recall_mean": 0.3, "cv_balanced_accuracy_mean": 0.3, "cv_geometric_mean_mean": 0.3},
        {"experiment_id": "d1__b__y", "dataset": "d1", "classifier": "b", "treatment": "y", "positive_label": 1, "execution_status": "success", "cv_average_precision_mean": 0.8, "cv_f1_mean": 0.3, "cv_recall_mean": 0.3, "cv_balanced_accuracy_mean": 0.3, "cv_geometric_mean_mean": 0.3},
        {"experiment_id": "d2__a__x", "dataset": "d2", "classifier": "a", "treatment": "x", "positive_label": 1, "execution_status": "success", "cv_average_precision_mean": 0.2, "cv_f1_mean": 0.3, "cv_recall_mean": 0.3, "cv_balanced_accuracy_mean": 0.3, "cv_geometric_mean_mean": 0.3},
        {"experiment_id": "d2__b__y", "dataset": "d2", "classifier": "b", "treatment": "y", "positive_label": 1, "execution_status": "success", "cv_average_precision_mean": 0.7, "cv_f1_mean": 0.3, "cv_recall_mean": 0.3, "cv_balanced_accuracy_mean": 0.3, "cv_geometric_mean_mean": 0.3},
    ])
    test = cv[["experiment_id", "dataset", "classifier", "treatment", "positive_label", "execution_status"]].copy()
    write_summary_tables(cv, test, tmp_path)
    within = pd.read_csv(tmp_path / "within_dataset_ranking.csv")
    overall = pd.read_csv(tmp_path / "overall_model_ranking.csv")
    assert "within_dataset_rank" in within.columns
    assert list(overall.columns).count("mean_within_dataset_rank") == 1
    assert "mean_cv_average_precision" in overall.columns
