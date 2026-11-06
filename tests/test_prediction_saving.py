import pandas as pd

from src.experiments import build_pipeline, create_experiment_id, make_train_test_split, prediction_frame, save_prediction_file
from src.evaluation import fit_predict_evaluate
from src.postrun_validation import validate_prediction_file


def test_prediction_file_matches_untouched_test_set(bundles, tmp_path):
    bundle = bundles["satimage"]
    X_train, X_test, y_train, y_test = make_train_test_split(bundle.X, bundle.y)
    pipeline = build_pipeline("logistic_regression", "original", y_train=y_train)
    _, y_pred, y_score = fit_predict_evaluate(pipeline, X_train, y_train, X_test, y_test)
    experiment_id = create_experiment_id(bundle.name, "logistic_regression", "original")
    predictions = prediction_frame(bundle, "logistic_regression", "original", experiment_id, X_test, y_test, y_pred, y_score)
    output = save_prediction_file(predictions, tmp_path)
    saved = pd.read_csv(output)

    assert len(saved) == len(y_test)
    assert not saved["row_index"].duplicated().any()
    assert saved["y_true"].value_counts().sort_index().to_dict() == y_test.value_counts().sort_index().to_dict()
    assert pd.api.types.is_numeric_dtype(saved["y_score"])
    validate_prediction_file(output, len(y_test), y_test.value_counts().sort_index().to_dict())
