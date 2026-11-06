from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_SPLITS = 5
PRIMARY_METRIC = "average_precision"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

REQUESTED_DATASETS = {
    "satimage": "satimage",
    "ozone_level": "ozone_level",
    "abalone_19": "abalone_19",
}

CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
TREATMENTS = (
    "original",
    "random_oversampling",
    "random_undersampling",
    "smote",
    "class_weighting",
)

PHASE1_SMOKE_CLASSIFIER = "logistic_regression"
PHASE1_SMOKE_TREATMENT = "original"
