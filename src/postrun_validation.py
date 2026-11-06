from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR


BOUNDED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f2",
    "roc_auc",
    "average_precision",
    "geometric_mean",
)

CV_METRICS = (
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
)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required output file is missing: {path}")
    return pd.read_csv(path)


def validate_prediction_file(path: Path, expected_test_count: int, expected_distribution: dict[int, int]) -> None:
    predictions = _read_required_csv(path)
    required = {"row_index", "y_true", "y_pred", "y_score", "dataset", "classifier", "treatment", "experiment_id"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if len(predictions) != expected_test_count:
        raise ValueError(f"{path} contains {len(predictions)} rows, expected {expected_test_count}")
    if predictions["row_index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_index values")
    actual_distribution = predictions["y_true"].value_counts().sort_index().to_dict()
    if actual_distribution != expected_distribution:
        raise ValueError(f"{path} y_true distribution {actual_distribution} != expected {expected_distribution}")
    if not pd.api.types.is_numeric_dtype(predictions["y_score"]):
        raise ValueError(f"{path} y_score must be numeric probability or decision-score values")


def _validate_metric_bounds(df: pd.DataFrame, metrics: tuple[str, ...], prefix: str = "") -> None:
    successful = df.loc[df["execution_status"] == "success"]
    for metric in metrics:
        column = f"{prefix}{metric}"
        if column not in successful.columns:
            raise ValueError(f"Missing required metric column {column}")
        if metric == "mcc":
            values = successful[column]
            if ((values < -1) | (values > 1)).any():
                raise ValueError(f"Metric {column} contains values outside [-1, 1]")
        else:
            values = successful[column]
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Metric {column} contains values outside [0, 1]")


def _expected_ranking(cv_results: pd.DataFrame) -> pd.DataFrame:
    successful_cv = cv_results.loc[cv_results["execution_status"] == "success"].copy()
    ranking_columns = [
        "dataset",
        "cv_average_precision_mean",
        "cv_f1_mean",
        "cv_recall_mean",
        "cv_balanced_accuracy_mean",
        "cv_geometric_mean_mean",
    ]
    within = successful_cv.sort_values(
        ranking_columns,
        ascending=[True, False, False, False, False, False],
    ).copy()
    within["within_dataset_rank"] = within.groupby("dataset").cumcount() + 1
    return within


def _compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], columns: list[str], name: str) -> None:
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if len(actual) != len(expected):
        raise ValueError(f"{name} row count {len(actual)} != expected {len(expected)}")
    for column in columns:
        if column not in actual.columns:
            raise ValueError(f"{name} is missing column {column}")
        if column not in expected.columns:
            raise ValueError(f"Expected {name} data is missing column {column}")
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            if not ((left.astype(float) - right.astype(float)).abs().fillna(0) < 1e-10).all():
                raise ValueError(f"{name} column {column} does not agree with raw results")
        elif left.astype(str).tolist() != right.astype(str).tolist():
            raise ValueError(f"{name} column {column} does not agree with raw results")


def validate_full_run_outputs(bundles, expected_grid: pd.DataFrame) -> None:
    expected_ids = set(expected_grid["experiment_id"])
    if len(expected_ids) != 45:
        raise ValueError(f"Expected full grid to contain 45 unique configurations, found {len(expected_ids)}")

    required_files = [
        METRICS_DIR / "cross_validation_results.csv",
        METRICS_DIR / "test_results.csv",
        METRICS_DIR / "all_experiment_results.csv",
        METRICS_DIR / "full_experiment_checkpoint.csv",
        METRICS_DIR / "experiment_manifest.csv",
        TABLES_DIR / "model_configurations.csv",
        TABLES_DIR / "best_model_per_dataset.csv",
        TABLES_DIR / "within_dataset_ranking.csv",
        TABLES_DIR / "overall_model_ranking.csv",
        TABLES_DIR / "average_performance_by_classifier.csv",
        TABLES_DIR / "average_performance_by_treatment.csv",
    ]
    for path in required_files:
        _read_required_csv(path)

    test_results = _read_required_csv(METRICS_DIR / "test_results.csv")
    cv_results = _read_required_csv(METRICS_DIR / "cross_validation_results.csv")
    manifest = _read_required_csv(METRICS_DIR / "experiment_manifest.csv")
    model_configurations = _read_required_csv(TABLES_DIR / "model_configurations.csv")

    for name, df in {"test_results": test_results, "cross_validation_results": cv_results}.items():
        if df["experiment_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate experiment IDs")
        missing = expected_ids.difference(set(df["experiment_id"]))
        if missing:
            raise ValueError(f"{name} is missing expected configurations: {sorted(missing)}")

    unexpected = set(test_results["experiment_id"]).difference(expected_ids)
    if unexpected:
        raise ValueError(f"Unexpected experiment IDs found: {sorted(unexpected)}")
    if set(cv_results["experiment_id"]) != expected_ids:
        raise ValueError("cross_validation_results.csv experiment IDs do not match the expected full grid")
    if set(model_configurations["experiment_id"]) != expected_ids:
        raise ValueError("model_configurations.csv experiment IDs do not match the expected full grid")

    for name, df in {"test_results": test_results, "cross_validation_results": cv_results}.items():
        statuses = set(df["execution_status"])
        if statuses != {"success"}:
            raise ValueError(f"{name} contains non-success statuses: {sorted(statuses)}")

    _validate_metric_bounds(test_results, BOUNDED_METRICS)
    _validate_metric_bounds(test_results, ("mcc",))

    for metric in CV_METRICS:
        mean_col = f"cv_{metric}_mean"
        std_col = f"cv_{metric}_std"
        if mean_col not in cv_results.columns or std_col not in cv_results.columns:
            raise ValueError(f"CV metric {metric} must have both mean and standard deviation columns")
        mean_values = cv_results.loc[cv_results["execution_status"] == "success", mean_col]
        if metric == "mcc":
            if ((mean_values < -1) | (mean_values > 1)).any():
                raise ValueError(f"CV metric {metric} contains mean values outside [-1, 1]")
        elif ((mean_values < 0) | (mean_values > 1)).any():
            raise ValueError(f"CV metric {metric} contains mean values outside [0, 1]")
        std_values = cv_results.loc[cv_results["execution_status"] == "success", std_col]
        if (std_values < 0).any():
            raise ValueError(f"CV metric {metric} contains negative standard deviations")

    successful = test_results.loc[test_results["execution_status"] == "success"]
    for row in successful.to_dict(orient="records"):
        bundle = bundles[row["dataset"]]
        _, X_test, _, y_test = __import__("src.experiments", fromlist=["make_train_test_split"]).make_train_test_split(bundle.X, bundle.y)
        expected_distribution = y_test.value_counts().sort_index().to_dict()
        prediction_path = PREDICTIONS_DIR / f"{row['experiment_id']}.csv.gz"
        validate_prediction_file(prediction_path, len(y_test), expected_distribution)
        predictions = pd.read_csv(prediction_path)
        for column in ("dataset", "classifier", "treatment", "experiment_id"):
            if set(predictions[column].unique()) != {row[column]}:
                raise ValueError(f"{prediction_path} has inconsistent {column}")
        if int(row["tn"] + row["fp"] + row["fn"] + row["tp"]) != len(y_test):
            raise ValueError(f"Confusion-matrix counts for {row['experiment_id']} do not sum to test-set size")

    if manifest["experiment_id"].duplicated().any():
        raise ValueError("experiment_manifest.csv contains duplicate experiment IDs")
    if set(manifest["experiment_id"]) != expected_ids:
        raise ValueError("experiment_manifest.csv experiment IDs do not match the expected full grid")
    if set(manifest["status"]) != {"complete"}:
        raise ValueError("experiment_manifest.csv contains non-complete statuses")
    for column in ("test_result_saved", "cv_result_saved", "prediction_file_saved", "artifacts_validated"):
        if not manifest[column].map(lambda value: str(value).lower() == "true").all():
            raise ValueError(f"experiment_manifest.csv column {column} is not true for every configuration")

    all_results = _read_required_csv(METRICS_DIR / "all_experiment_results.csv")
    checkpoint = _read_required_csv(METRICS_DIR / "full_experiment_checkpoint.csv")
    if set(all_results["experiment_id"]) != expected_ids:
        raise ValueError("all_experiment_results.csv experiment IDs do not match the expected full grid")
    if set(checkpoint["experiment_id"]) != expected_ids:
        raise ValueError("full_experiment_checkpoint.csv experiment IDs do not match the expected full grid")

    expected_within = _expected_ranking(cv_results)
    within = _read_required_csv(TABLES_DIR / "within_dataset_ranking.csv")
    _compare_frames(
        within,
        expected_within,
        keys=["dataset", "within_dataset_rank"],
        columns=[
            "experiment_id",
            "dataset",
            "classifier",
            "treatment",
            "cv_average_precision_mean",
            "cv_f1_mean",
            "cv_recall_mean",
            "cv_balanced_accuracy_mean",
            "cv_geometric_mean_mean",
            "within_dataset_rank",
        ],
        name="within_dataset_ranking.csv",
    )

    expected_best = expected_within.groupby("dataset", as_index=False).head(1)
    best = _read_required_csv(TABLES_DIR / "best_model_per_dataset.csv")
    if "cv_average_precision_mean" not in best.columns:
        raise ValueError("best_model_per_dataset.csv must include CV Average Precision selection evidence")
    if set(best["experiment_id"]) != set(expected_best["experiment_id"]):
        raise ValueError("best_model_per_dataset.csv does not match CV-selected best configurations")

    expected_overall = (
        expected_within
        .groupby(["classifier", "treatment"], as_index=False)
        .agg(
            mean_within_dataset_rank=("within_dataset_rank", "mean"),
            std_within_dataset_rank=("within_dataset_rank", "std"),
            median_within_dataset_rank=("within_dataset_rank", "median"),
            mean_cv_average_precision=("cv_average_precision_mean", "mean"),
            mean_cv_f1=("cv_f1_mean", "mean"),
            mean_cv_recall=("cv_recall_mean", "mean"),
            mean_cv_balanced_accuracy=("cv_balanced_accuracy_mean", "mean"),
            mean_cv_geometric_mean=("cv_geometric_mean_mean", "mean"),
            datasets_evaluated=("dataset", "nunique"),
        )
        .sort_values(["mean_within_dataset_rank", "mean_cv_average_precision", "mean_cv_f1"], ascending=[True, False, False])
    )
    overall = _read_required_csv(TABLES_DIR / "overall_model_ranking.csv")
    _compare_frames(
        overall,
        expected_overall,
        keys=["mean_within_dataset_rank", "classifier", "treatment"],
        columns=[
            "classifier",
            "treatment",
            "mean_within_dataset_rank",
            "std_within_dataset_rank",
            "median_within_dataset_rank",
            "mean_cv_average_precision",
            "mean_cv_f1",
            "mean_cv_recall",
            "mean_cv_balanced_accuracy",
            "mean_cv_geometric_mean",
            "datasets_evaluated",
        ],
        name="overall_model_ranking.csv",
    )
