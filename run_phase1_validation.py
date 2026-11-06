from __future__ import annotations

from src.config import LOGS_DIR, METRICS_DIR
from src.experiments import run_phase1_smoke_tests
from src.validation import configure_logger, load_validate_and_write_tables, write_software_versions


def main() -> None:
    logger = configure_logger(LOGS_DIR / "phase1_validation.log")
    logger.info("starting Phase 1 validation")
    versions = write_software_versions(LOGS_DIR / "software_versions.txt")
    logger.info("recorded software versions: %s", versions)
    bundles, summary, mappings = load_validate_and_write_tables(logger)
    logger.info("loaded datasets: %s", summary[["dataset", "observations", "input_features"]].to_dict(orient="records"))
    logger.info("label mappings: %s", mappings.to_dict(orient="records"))
    smoke_results = run_phase1_smoke_tests(bundles, METRICS_DIR / "smoke_test_results.csv", logger=logger)
    logger.info("smoke-test statuses: %s", smoke_results[["experiment_id", "execution_status"]].to_dict(orient="records"))
    logger.info("Phase 1 validation complete")


if __name__ == "__main__":
    main()

