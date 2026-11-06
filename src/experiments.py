from __future__ import annotations

import os
import logging
from pathlib import Path
import tempfile
from typing import Iterable
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import (
    CLASSIFIERS,
    CV_SPLITS,
    METRICS_DIR,
    POSITIVE_LABEL,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TEST_SIZE,
    TREATMENTS,
)
from .evaluation import CV_SCORERS, fit_predict_evaluate
from .models import get_model
from .preprocessing import get_preprocessor
from .samplers import get_sampler


def create_experiment_id(dataset: str, classifier: str, treatment: str) -> str:
    return f"{dataset}__{classifier}__{treatment}"


def build_full_experiment_grid(
    datasets: Iterable[str],
    classifiers: Iterable[str] = CLASSIFIERS,
    treatments: Iterable[str] = TREATMENTS,
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for classifier in classifiers:
            for treatment in treatments:
                rows.append({
                    "experiment_id": create_experiment_id(dataset, classifier, treatment),
                    "dataset": dataset,
                    "classifier": classifier,
                    "treatment": treatment,
                })
    grid = pd.DataFrame(rows)
    if grid["experiment_id"].duplicated().any():
        duplicates = grid.loc[grid["experiment_id"].duplicated(), "experiment_id"].tolist()
        raise ValueError(f"Duplicate experiment IDs are not allowed: {duplicates}")
    return grid


def make_train_test_split(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline(classifier_name: str, treatment: str, y_train=None) -> Pipeline:
    steps = [("preprocess", get_preprocessor(classifier_name, treatment))]
    sampler = get_sampler(treatment)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", get_model(classifier_name, treatment=treatment, y_train=y_train)))
    return Pipeline(steps)


def make_cv(n_splits: int = CV_SPLITS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def _load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "experiment_id" in df.columns and df["experiment_id"].duplicated().any():
        duplicates = df.loc[df["experiment_id"].duplicated(), "experiment_id"].tolist()
        raise ValueError(f"Existing results contain duplicate experiment IDs: {duplicates}")
    return df


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output_path)
    except Exception:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def save_result_row(row: dict, output_path: Path, allow_success_overwrite: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_results(output_path)
    experiment_id = row["experiment_id"]

    if not existing.empty and experiment_id in set(existing["experiment_id"]):
        previous = existing.loc[existing["experiment_id"] == experiment_id].iloc[0]
        if previous.get("execution_status") == "success" and not allow_success_overwrite:
            raise ValueError(f"Refusing to overwrite completed experiment {experiment_id}")
        existing = existing.loc[existing["experiment_id"] != experiment_id]

    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    if updated["experiment_id"].duplicated().any():
        raise ValueError("Duplicate experiment IDs detected while saving results")
    _atomic_write_csv(updated, output_path)


def _manifest_columns() -> list[str]:
    return [
        "experiment_id",
        "dataset",
        "classifier",
        "treatment",
        "status",
        "test_result_saved",
        "cv_result_saved",
        "prediction_file_saved",
        "artifacts_validated",
        "started_at",
        "completed_at",
        "warning_or_error_message",
    ]


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=_manifest_columns())


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_manifest()
    df = pd.read_csv(path)
    if df["experiment_id"].duplicated().any():
        duplicates = df.loc[df["experiment_id"].duplicated(), "experiment_id"].tolist()
        raise ValueError(f"Manifest contains duplicate experiment IDs: {duplicates}")
    return df


def save_manifest_row(row: dict, manifest_path: Path) -> None:
    allowed = {"pending", "running", "failed", "complete"}
    if row["status"] not in allowed:
        raise ValueError(f"Invalid manifest status {row['status']!r}")
    existing = _load_manifest(manifest_path)
    updated = existing.loc[existing["experiment_id"] != row["experiment_id"]]
    updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
    if updated["experiment_id"].duplicated().any():
        raise ValueError("Duplicate experiment IDs detected while saving manifest")
    _atomic_write_csv(updated, manifest_path)


def initialize_manifest(path: Path, overwrite: bool = False) -> None:
    if overwrite or not path.exists():
        _atomic_write_csv(_empty_manifest(), path)


def result_row_is_success(path: Path, experiment_id: str) -> bool:
    df = _load_existing_results(path)
    if df.empty or experiment_id not in set(df["experiment_id"]):
        return False
    row = df.loc[df["experiment_id"] == experiment_id].iloc[0]
    return row.get("execution_status") == "success"


def expected_test_distribution(bundle) -> tuple[int, dict[int, int]]:
    _, X_test, _, y_test = make_train_test_split(bundle.X, bundle.y)
    return len(X_test), y_test.value_counts().sort_index().to_dict()


def artifacts_are_complete_and_valid(
    bundle,
    config: dict,
    test_output_path: Path,
    cv_output_path: Path,
    prediction_output_dir: Path,
) -> bool:
    experiment_id = config["experiment_id"]
    if not result_row_is_success(test_output_path, experiment_id):
        return False
    if not result_row_is_success(cv_output_path, experiment_id):
        return False
    prediction_path = prediction_output_dir / f"{experiment_id}.csv.gz"
    if not prediction_path.exists():
        return False
    expected_count, expected_distribution = expected_test_distribution(bundle)
    from .postrun_validation import validate_prediction_file

    validate_prediction_file(prediction_path, expected_count, expected_distribution)
    predictions = pd.read_csv(prediction_path)
    identifiers = {
        "dataset": config["dataset"],
        "classifier": config["classifier"],
        "treatment": config["treatment"],
        "experiment_id": experiment_id,
    }
    for column, expected in identifiers.items():
        if set(predictions[column].unique()) != {expected}:
            raise ValueError(f"Prediction file {prediction_path} has inconsistent {column}")
    return True


def manifest_marks_complete(manifest_path: Path, experiment_id: str) -> bool:
    manifest = _load_manifest(manifest_path)
    if manifest.empty or experiment_id not in set(manifest["experiment_id"]):
        return False
    row = manifest.loc[manifest["experiment_id"] == experiment_id].iloc[0]
    return row.get("status") == "complete" and bool(row.get("artifacts_validated"))


def experiment_is_complete(
    bundle,
    config: dict,
    manifest_path: Path,
    test_output_path: Path,
    cv_output_path: Path,
    prediction_output_dir: Path,
) -> bool:
    return (
        manifest_marks_complete(manifest_path, config["experiment_id"])
        and artifacts_are_complete_and_valid(bundle, config, test_output_path, cv_output_path, prediction_output_dir)
    )


def make_manifest_row(
    config: dict,
    status: str,
    started_at: str,
    completed_at: str = "",
    test_result_saved: bool = False,
    cv_result_saved: bool = False,
    prediction_file_saved: bool = False,
    artifacts_validated: bool = False,
    message: str = "",
) -> dict:
    return {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "classifier": config["classifier"],
        "treatment": config["treatment"],
        "status": status,
        "test_result_saved": bool(test_result_saved),
        "cv_result_saved": bool(cv_result_saved),
        "prediction_file_saved": bool(prediction_file_saved),
        "artifacts_validated": bool(artifacts_validated),
        "started_at": started_at,
        "completed_at": completed_at,
        "warning_or_error_message": message,
    }


def cross_validate_on_training_data(pipeline, X_train, y_train, n_splits: int = CV_SPLITS) -> dict[str, float]:
    cv = make_cv(n_splits=n_splits)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=CV_SCORERS,
        n_jobs=None,
        error_score="raise",
    )
    result = {"cv_splits": n_splits}
    for metric in CV_SCORERS:
        values = scores[f"test_{metric}"]
        result[f"cv_{metric}_mean"] = float(np.mean(values))
        result[f"cv_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def prediction_frame(bundle, classifier: str, treatment: str, experiment_id: str, X_test, y_test, y_pred, y_score) -> pd.DataFrame:
    return pd.DataFrame({
        "row_index": X_test.index.to_numpy(),
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "y_score": np.asarray(y_score),
        "dataset": bundle.name,
        "classifier": classifier,
        "treatment": treatment,
        "experiment_id": experiment_id,
    })


def fitted_scale_pos_weight(pipeline) -> float | None:
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    value = getattr(model, "scale_pos_weight_", None)
    return None if value is None else float(value)


def save_prediction_file(predictions: pd.DataFrame, output_dir: Path = PREDICTIONS_DIR) -> Path:
    if predictions["row_index"].duplicated().any():
        raise ValueError("Prediction file contains duplicate row indices")
    experiment_id = predictions["experiment_id"].iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_id}.csv.gz"
    tmp_path = output_dir / f".{experiment_id}.csv.gz.tmp"
    try:
        predictions.to_csv(tmp_path, index=False, compression="gzip")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return output_path


def run_single_experiment(bundle, classifier: str, treatment: str, cv_splits: int = CV_SPLITS) -> tuple[dict, dict, pd.DataFrame | None]:
    X_train, X_test, y_train, y_test = make_train_test_split(bundle.X, bundle.y)
    pipeline = build_pipeline(classifier, treatment, y_train=y_train)
    experiment_id = create_experiment_id(bundle.name, classifier, treatment)
    row = {
        "experiment_id": experiment_id,
        "dataset": bundle.name,
        "classifier": classifier,
        "treatment": treatment,
        "positive_label": POSITIVE_LABEL,
        "test_sample_count": int(len(y_test)),
        "train_sample_count": int(len(y_train)),
    }
    cv_row = {
        "experiment_id": experiment_id,
        "dataset": bundle.name,
        "classifier": classifier,
        "treatment": treatment,
        "positive_label": POSITIVE_LABEL,
        "execution_status": "pending",
        "warning_or_error_message": "",
    }
    prediction_rows = None
    captured_messages = []
    try:
        if not bundle.metadata.get("modeling_ready", True):
            raise ValueError(f"Dataset {bundle.name} is not modeling-ready: {bundle.cleaning_metadata['cleaning_status']}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if cv_splits > 1:
                cv_row.update(cross_validate_on_training_data(pipeline, X_train, y_train, n_splits=cv_splits))
            metrics, y_pred, y_score = fit_predict_evaluate(pipeline, X_train, y_train, X_test, y_test)
        captured_messages = [str(item.message) for item in caught]
        row.update(metrics)
        row["scale_pos_weight"] = fitted_scale_pos_weight(pipeline)
        row["execution_status"] = "success"
        row["warning_or_error_message"] = " | ".join(captured_messages)
        cv_row["scale_pos_weight"] = row["scale_pos_weight"]
        cv_row["execution_status"] = "success"
        cv_row["warning_or_error_message"] = row["warning_or_error_message"]
        prediction_rows = prediction_frame(bundle, classifier, treatment, experiment_id, X_test, y_test, y_pred, y_score)
    except Exception as exc:
        row["execution_status"] = "failed"
        row["warning_or_error_message"] = repr(exc)
        cv_row["execution_status"] = "failed"
        cv_row["warning_or_error_message"] = repr(exc)
    return row, cv_row, prediction_rows


def completed_experiment_ids(path: Path) -> set[str]:
    existing = _load_existing_results(path)
    if existing.empty:
        return set()
    return set(existing.loc[existing["execution_status"] == "success", "experiment_id"])


def run_configurations(
    bundles,
    grid: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger | None = None,
    cv_splits: int = CV_SPLITS,
    cv_output_path: Path | None = None,
    prediction_output_dir: Path | None = None,
    save_predictions: bool = False,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    prediction_output_dir = prediction_output_dir or PREDICTIONS_DIR
    cv_output_path = cv_output_path or ((METRICS_DIR / "cross_validation_results.csv") if save_predictions else None)
    use_manifest = save_predictions or manifest_path is not None
    manifest_path = manifest_path or (METRICS_DIR / "experiment_manifest.csv")
    completed_without_manifest = completed_experiment_ids(output_path) if not save_predictions else set()

    for config in grid.to_dict(orient="records"):
        experiment_id = config["experiment_id"]
        bundle = bundles[config["dataset"]]
        if save_predictions and experiment_is_complete(bundle, config, manifest_path, output_path, cv_output_path, prediction_output_dir):
            logger.info("skipped completed configuration %s", experiment_id)
            continue
        if not save_predictions and experiment_id in completed_without_manifest:
            logger.info("skipped completed configuration %s", experiment_id)
            continue
        logger.info("running configuration %s", experiment_id)
        started_at = utc_now_iso()
        if use_manifest:
            save_manifest_row(make_manifest_row(config, "running", started_at), manifest_path)
        row, cv_row, predictions = run_single_experiment(
            bundle,
            config["classifier"],
            config["treatment"],
            cv_splits=cv_splits,
        )
        if row["execution_status"] != "success":
            if use_manifest:
                save_manifest_row(make_manifest_row(config, "failed", started_at, completed_at=utc_now_iso(), message=row["warning_or_error_message"]), manifest_path)
            logger.error("failed configuration %s: %s", experiment_id, row["warning_or_error_message"])
            continue

        prediction_saved = False
        cv_saved = False
        test_saved = False
        try:
            if save_predictions:
                if predictions is None:
                    raise ValueError(f"Successful experiment {experiment_id} did not produce predictions")
                path = save_prediction_file(predictions, prediction_output_dir)
                prediction_saved = True
                logger.info("saved predictions for %s to %s", experiment_id, path)
            if cv_output_path is not None:
                save_result_row(cv_row, cv_output_path, allow_success_overwrite=True)
                cv_saved = True
            save_result_row(row, output_path, allow_success_overwrite=True)
            test_saved = True
            artifacts_validated = True
            if save_predictions:
                artifacts_validated = artifacts_are_complete_and_valid(bundle, config, output_path, cv_output_path, prediction_output_dir)
            if use_manifest:
                save_manifest_row(make_manifest_row(
                    config,
                    "complete",
                    started_at,
                    completed_at=utc_now_iso(),
                    test_result_saved=test_saved,
                    cv_result_saved=cv_saved,
                    prediction_file_saved=prediction_saved,
                    artifacts_validated=artifacts_validated,
                    message=row["warning_or_error_message"],
                ), manifest_path)
            logger.info("successful configuration %s warnings=%s", experiment_id, row["warning_or_error_message"])
        except Exception as exc:
            if use_manifest:
                save_manifest_row(make_manifest_row(
                    config,
                    "failed",
                    started_at,
                    completed_at=utc_now_iso(),
                    test_result_saved=test_saved,
                    cv_result_saved=cv_saved,
                    prediction_file_saved=prediction_saved,
                    artifacts_validated=False,
                    message=repr(exc),
                ), manifest_path)
            logger.error("failed artifact save for %s: %r", experiment_id, exc)
            raise

    return pd.read_csv(output_path)


def run_phase1_smoke_tests(
    bundles,
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
    cv_output_path: Path | None = None,
    manifest_path: Path | None = None,
    save_predictions: bool = False,
    prediction_output_dir: Path | None = None,
) -> pd.DataFrame:
    output_path = output_path or (METRICS_DIR / "smoke_test_results.csv")
    grid = build_full_experiment_grid(
        datasets=bundles.keys(),
        classifiers=("logistic_regression",),
        treatments=("original",),
    )
    return run_configurations(
        bundles,
        grid,
        output_path,
        logger=logger,
        cv_splits=2,
        cv_output_path=cv_output_path,
        manifest_path=manifest_path,
        save_predictions=save_predictions,
        prediction_output_dir=prediction_output_dir,
    )


def write_model_configuration_table(grid: pd.DataFrame, output_path: Path = TABLES_DIR / "model_configurations.csv") -> None:
    _atomic_write_csv(grid, output_path)


def write_summary_tables(cv_results: pd.DataFrame, test_results: pd.DataFrame, tables_dir: Path = TABLES_DIR) -> None:
    successful_cv = cv_results.loc[cv_results["execution_status"] == "success"].copy()
    successful_test = test_results.loc[test_results["execution_status"] == "success"].copy()
    if successful_cv.empty or successful_test.empty:
        return

    ranking = successful_cv.sort_values(
        ["dataset", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"],
        ascending=[True, False, False, False, False, False],
    )
    best = ranking.groupby("dataset", as_index=False).head(1)
    best = best.merge(
        successful_test,
        on=["experiment_id", "dataset", "classifier", "treatment", "positive_label"],
        how="left",
        suffixes=("_cv", "_test"),
    )

    overall = successful_cv.sort_values(
        ["cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"],
        ascending=False,
    )
    within_dataset = successful_cv.sort_values(
        ["dataset", "cv_average_precision_mean", "cv_f1_mean", "cv_recall_mean", "cv_balanced_accuracy_mean", "cv_geometric_mean_mean"],
        ascending=[True, False, False, False, False, False],
    ).copy()
    within_dataset["within_dataset_rank"] = within_dataset.groupby("dataset").cumcount() + 1
    overall = (
        within_dataset
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
    by_classifier = successful_cv.groupby("classifier", as_index=False).mean(numeric_only=True)
    by_treatment = successful_cv.groupby("treatment", as_index=False).mean(numeric_only=True)

    _atomic_write_csv(best, tables_dir / "best_model_per_dataset.csv")
    _atomic_write_csv(within_dataset, tables_dir / "within_dataset_ranking.csv")
    _atomic_write_csv(overall, tables_dir / "overall_model_ranking.csv")
    _atomic_write_csv(by_classifier, tables_dir / "average_performance_by_classifier.csv")
    _atomic_write_csv(by_treatment, tables_dir / "average_performance_by_treatment.csv")
