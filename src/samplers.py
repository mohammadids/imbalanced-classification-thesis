from __future__ import annotations

from collections import Counter
import warnings

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator

from .config import RANDOM_STATE


class AdaptiveSMOTE(BaseEstimator):
    """SMOTE wrapper that adjusts k_neighbors to each training fold."""

    def __init__(self, sampling_strategy="auto", random_state=RANDOM_STATE, k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        counts = Counter(y)
        minority_count = min(counts.values())
        if minority_count < 2:
            raise ValueError("SMOTE requires at least two minority samples in the training data")
        actual_k = min(self.k_neighbors, minority_count - 1)
        if actual_k != self.k_neighbors:
            warnings.warn(
                f"Adjusted SMOTE k_neighbors from {self.k_neighbors} to {actual_k} "
                f"because the training fold has {minority_count} minority samples.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.actual_k_neighbors_ = actual_k
        self.sampler_ = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=actual_k,
        )
        return self.sampler_.fit_resample(X, y)


def get_sampler(treatment: str):
    if treatment == "original" or treatment == "class_weighting":
        return None
    if treatment == "random_oversampling":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if treatment == "random_undersampling":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    if treatment == "smote":
        return AdaptiveSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    raise ValueError(f"Unsupported treatment {treatment!r}")

