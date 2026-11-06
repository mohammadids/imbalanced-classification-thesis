from __future__ import annotations

import pandas as pd

from src.config import LOGS_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR
from src.data_loader import load_all_datasets
from src.experiments import (
    _atomic_write_csv,
    build_full_experiment_grid,
    run_configurations,
    write_model_configuration_table,
    write_summary_tables,
)
from src.postrun_validation import validate_full_run_outputs
from src.validation import configure_logger, validate_bundles, write_software_versions


def main() -> None:
    logger = configure_logger(LOGS_DIR / "full_experiment.log", logger_name="full_experiment", mode="a")
    logger.info("starting full experiment matrix")
    versions = write_software_versions(LOGS_DIR / "software_versions.txt")
    logger.info("recorded software versions: %s", versions)
    bundles = load_all_datasets()
    validate_bundles(bundles, require_modeling_ready=True)
    grid = build_full_experiment_grid(bundles.keys())
    write_model_configuration_table(grid, TABLES_DIR / "model_configurations.csv")
    test_results = run_configurations(
        bundles,
        grid,
        METRICS_DIR / "test_results.csv",
        logger=logger,
        cv_output_path=METRICS_DIR / "cross_validation_results.csv",
        prediction_output_dir=PREDICTIONS_DIR,
        save_predictions=True,
        manifest_path=METRICS_DIR / "experiment_manifest.csv",
    )
    cv_results = pd.read_csv(METRICS_DIR / "cross_validation_results.csv")
    write_summary_tables(cv_results, test_results, TABLES_DIR)
    all_results = cv_results.merge(
        test_results,
        on=["experiment_id", "dataset", "classifier", "treatment", "positive_label", "execution_status", "warning_or_error_message"],
        how="outer",
        suffixes=("_cv", "_test"),
    )
    _atomic_write_csv(all_results, METRICS_DIR / "all_experiment_results.csv")
    _atomic_write_csv(test_results, METRICS_DIR / "full_experiment_checkpoint.csv")
    validate_full_run_outputs(bundles, grid)
    logger.info("full experiment matrix complete")


if __name__ == "__main__":
    main()
