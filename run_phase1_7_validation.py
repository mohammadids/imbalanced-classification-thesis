from __future__ import annotations

from sklearn.base import is_classifier
from sklearn.utils import get_tags

from src.config import LOGS_DIR, METRICS_DIR, REQUESTED_DATASETS
from src.data_loader import load_all_datasets
from src.experiments import build_full_experiment_grid, run_configurations
from src.models import FoldAwareXGBClassifier
from src.validation import configure_logger, validate_bundles


def main() -> None:
    logger = configure_logger(LOGS_DIR / "phase1_7_validation.log", logger_name="phase1_7_validation", mode="w")
    model = FoldAwareXGBClassifier()
    logger.info("FoldAwareXGBClassifier is_classifier=%s", is_classifier(model))
    logger.info("FoldAwareXGBClassifier estimator_type=%s", get_tags(model).estimator_type)

    logger.info("loading and validating final datasets without rewriting previous phase tables")
    bundles = load_all_datasets()
    validate_bundles(bundles, require_modeling_ready=True)
    grid = build_full_experiment_grid(
        datasets=REQUESTED_DATASETS.keys(),
        classifiers=("xgboost",),
        treatments=("original", "class_weighting"),
    )
    logger.info("running %s Phase 1.7 XGBoost smoke configurations", len(grid))
    results = run_configurations(
        bundles,
        grid,
        METRICS_DIR / "xgboost_smoke_test_results_phase1_7.csv",
        logger=logger,
        cv_splits=2,
        cv_output_path=METRICS_DIR / "xgboost_smoke_cv_results_phase1_7.csv",
        save_predictions=False,
    )
    cv_results_path = METRICS_DIR / "xgboost_smoke_cv_results_phase1_7.csv"
    logger.info("test statuses: %s", results[["experiment_id", "execution_status"]].to_dict(orient="records"))
    logger.info("cv results saved to %s", cv_results_path)
    if not (results["execution_status"] == "success").all():
        raise RuntimeError("At least one Phase 1.7 XGBoost smoke test failed")
    logger.info("Phase 1.7 validation complete")


if __name__ == "__main__":
    main()
