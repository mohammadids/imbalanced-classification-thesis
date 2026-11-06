from __future__ import annotations

from src.config import LOGS_DIR, METRICS_DIR
from src.experiments import initialize_manifest, run_phase1_smoke_tests
from src.validation import (
    configure_logger,
    load_validate_and_write_phase1_6_tables,
    write_software_versions,
)


def main() -> None:
    logger = configure_logger(LOGS_DIR / "phase1_6_validation.log", logger_name="phase1_6_validation", mode="w")
    logger.info("starting Phase 1.6 validation")
    versions = write_software_versions(LOGS_DIR / "software_versions_phase1_6.txt")
    logger.info("recorded software versions: %s", versions)
    bundles, raw_summary, modeling_summary, cleaning_summary, mappings = load_validate_and_write_phase1_6_tables(logger)
    logger.info("raw datasets: %s", raw_summary[["dataset", "observations", "input_features"]].to_dict(orient="records"))
    logger.info("modeling datasets: %s", modeling_summary[["dataset", "observations", "input_features"]].to_dict(orient="records"))
    logger.info("cleaning summary: %s", cleaning_summary.to_dict(orient="records"))
    logger.info("label mappings: %s", mappings.to_dict(orient="records"))
    initialize_manifest(METRICS_DIR / "experiment_manifest.csv", overwrite=True)
    smoke_results = run_phase1_smoke_tests(
        bundles,
        METRICS_DIR / "smoke_test_results_phase1_6.csv",
        logger=logger,
        cv_output_path=METRICS_DIR / "smoke_cross_validation_results_phase1_6.csv",
    )
    logger.info("Phase 1.6 smoke-test statuses: %s", smoke_results[["experiment_id", "execution_status"]].to_dict(orient="records"))
    logger.info("Phase 1.6 validation complete")


if __name__ == "__main__":
    main()
