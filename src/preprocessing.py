from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def get_preprocessor(classifier_name: str, treatment: str = "original"):
    """Return a preprocessing step suitable for the classifier family."""
    if classifier_name == "logistic_regression" or treatment == "smote":
        return StandardScaler()
    return "passthrough"
