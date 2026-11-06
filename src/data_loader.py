from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets

from .config import POSITIVE_LABEL, NEGATIVE_LABEL, REQUESTED_DATASETS


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    fetch_key: str
    X: pd.DataFrame
    y: pd.Series
    original_y: pd.Series
    metadata: dict[str, Any]
    raw_X: pd.DataFrame
    raw_original_y: pd.Series
    raw_metadata: dict[str, Any]
    cleaning_metadata: dict[str, Any]


class DatasetCleaningError(ValueError):
    """Raised when duplicate feature vectors have conflicting target labels."""


def get_available_dataset_keys() -> list[str]:
    """Return requested dataset keys accepted by the installed imbalanced-learn package."""
    return list(
        fetch_datasets(
            filter_data=tuple(REQUESTED_DATASETS.values()),
            download_if_missing=True,
            verbose=False,
        ).keys()
    )


def fetch_requested_raw_datasets() -> dict[str, Any]:
    requested_keys = tuple(REQUESTED_DATASETS.values())
    available = fetch_datasets(filter_data=requested_keys, download_if_missing=True, verbose=False)
    missing = {
        name: key
        for name, key in REQUESTED_DATASETS.items()
        if key not in available
    }
    if missing:
        returned = ", ".join(available.keys())
        raise KeyError(f"Missing requested dataset keys {missing}. Returned keys: {returned}")
    return {name: available[key] for name, key in REQUESTED_DATASETS.items()}


def _feature_names(n_features: int) -> list[str]:
    return [f"feature_{idx:02d}" for idx in range(n_features)]


def _native_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _label_counts(y: pd.Series) -> pd.Series:
    return y.value_counts(dropna=False).sort_values(ascending=False)


def _class_metadata(name: str, fetch_key: str, X: pd.DataFrame, original_y: pd.Series, y: pd.Series, mapping: dict) -> dict[str, Any]:
    counts = _label_counts(original_y)
    combined = X.copy()
    combined["target"] = y
    missing_values = int(X.isna().sum().sum() + original_y.isna().sum())
    duplicate_observations = int(combined.duplicated().sum())
    majority_label = _native_label(counts.index[0])
    minority_label = _native_label(counts.index[1])
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[1])

    return {
        "dataset_name": name,
        "fetch_key": fetch_key,
        "n_observations": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "original_class_labels": [_native_label(label) for label in counts.index.tolist()],
        "majority_class_label": majority_label,
        "minority_class_label": minority_label,
        "majority_class_count": majority_count,
        "minority_class_count": minority_count,
        "minority_class_percentage": minority_count / int(X.shape[0]) * 100,
        "imbalance_ratio": majority_count / minority_count,
        "missing_values": missing_values,
        "duplicate_observations": duplicate_observations,
        "label_mapping": mapping,
        "descriptive_statistics": X.describe().to_dict(),
    }


def _detect_conflicting_feature_groups(X: pd.DataFrame, original_y: pd.Series) -> list[dict[str, Any]]:
    combined = X.copy()
    combined["original_target"] = original_y
    conflicts = []
    grouped = combined.groupby(list(X.columns), dropna=False, sort=False)["original_target"]
    for feature_values, labels in grouped:
        unique_labels = sorted(_native_label(label) for label in labels.unique().tolist())
        if len(unique_labels) > 1:
            indices = labels.index.tolist()
            conflicts.append({
                "row_indices": indices,
                "labels": unique_labels,
                "feature_values": dict(zip(X.columns, feature_values if isinstance(feature_values, tuple) else (feature_values,))),
            })
    return conflicts


def _clean_duplicate_rows(
    name: str,
    X: pd.DataFrame,
    original_y: pd.Series,
    y: pd.Series,
    raw_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    conflicts = _detect_conflicting_feature_groups(X, original_y)
    if conflicts:
        raw_counts = y.value_counts().sort_index()
        raw_majority_count = int(raw_counts.get(NEGATIVE_LABEL, 0))
        raw_minority_count = int(raw_counts.get(POSITIVE_LABEL, 0))
        cleaning_metadata = {
            "dataset": name,
            "raw_observations": int(len(y)),
            "cleaned_observations": None,
            "removed_duplicate_rows": 0,
            "removed_majority_rows": 0,
            "removed_minority_rows": 0,
            "raw_majority_count": raw_majority_count,
            "raw_minority_count": raw_minority_count,
            "cleaned_majority_count": None,
            "cleaned_minority_count": None,
            "raw_imbalance_ratio": raw_majority_count / raw_minority_count,
            "cleaned_imbalance_ratio": None,
            "duplicate_feature_rows": int(X.duplicated(keep=False).sum()),
            "exact_duplicate_feature_target_rows": int(pd.concat([X, y], axis=1).duplicated().sum()),
            "conflicting_feature_groups": len(conflicts),
            "conflicting_groups_json": json.dumps(conflicts, sort_keys=True),
            "cleaning_status": "blocked_conflicting_labels",
        }
        return X, original_y, y, cleaning_metadata

    combined = X.copy()
    combined["original_target"] = original_y
    combined["target"] = y
    duplicate_mask = combined.duplicated(subset=list(X.columns) + ["target"], keep="first")
    removed = combined.loc[duplicate_mask]
    cleaned = combined.loc[~duplicate_mask].copy()

    cleaned_X = cleaned.loc[:, X.columns]
    cleaned_original_y = cleaned["original_target"].rename("original_target")
    cleaned_y = cleaned["target"].astype(int).rename("target")

    cleaned_counts = cleaned_y.value_counts().sort_index()
    raw_counts = y.value_counts().sort_index()
    raw_majority_count = int(raw_counts.get(NEGATIVE_LABEL, 0))
    raw_minority_count = int(raw_counts.get(POSITIVE_LABEL, 0))
    cleaned_majority_count = int(cleaned_counts.get(NEGATIVE_LABEL, 0))
    cleaned_minority_count = int(cleaned_counts.get(POSITIVE_LABEL, 0))

    cleaning_metadata = {
        "dataset": name,
        "raw_observations": int(len(y)),
        "cleaned_observations": int(len(cleaned_y)),
        "removed_duplicate_rows": int(duplicate_mask.sum()),
        "removed_majority_rows": int((removed["target"] == NEGATIVE_LABEL).sum()),
        "removed_minority_rows": int((removed["target"] == POSITIVE_LABEL).sum()),
        "raw_majority_count": raw_majority_count,
        "raw_minority_count": raw_minority_count,
        "cleaned_majority_count": cleaned_majority_count,
        "cleaned_minority_count": cleaned_minority_count,
        "raw_imbalance_ratio": raw_majority_count / raw_minority_count,
        "cleaned_imbalance_ratio": cleaned_majority_count / cleaned_minority_count,
        "duplicate_feature_rows": int(X.duplicated(keep=False).sum()),
        "exact_duplicate_feature_target_rows": int(duplicate_mask.sum()),
        "conflicting_feature_groups": len(conflicts),
        "conflicting_groups_json": json.dumps(conflicts, sort_keys=True),
        "cleaning_status": "cleaned" if duplicate_mask.any() else "unchanged",
    }
    return cleaned_X, cleaned_original_y, cleaned_y, cleaning_metadata


def inspect_and_remap_dataset(name: str, fetch_key: str, raw_dataset: Any) -> DatasetBundle:
    X = pd.DataFrame(raw_dataset.data, columns=_feature_names(raw_dataset.data.shape[1]))
    original_y = pd.Series(raw_dataset.target, name="original_target")
    counts = _label_counts(original_y)

    if len(counts) != 2:
        raise ValueError(f"{name} must be binary, found labels {list(counts.index)}")
    if counts.iloc[0] == counts.iloc[1]:
        raise ValueError(f"{name} has tied class counts; majority/minority mapping is ambiguous")

    majority_label = _native_label(counts.index[0])
    minority_label = _native_label(counts.index[1])
    mapping = {majority_label: NEGATIVE_LABEL, minority_label: POSITIVE_LABEL}
    y = original_y.map(mapping).astype(int).rename("target")

    raw_metadata = _class_metadata(name, fetch_key, X, original_y, y, mapping)
    cleaned_X, cleaned_original_y, cleaned_y, cleaning_metadata = _clean_duplicate_rows(
        name, X, original_y, y, raw_metadata
    )
    modeling_metadata = _class_metadata(name, fetch_key, cleaned_X, cleaned_original_y, cleaned_y, mapping)
    modeling_metadata["raw_observations"] = raw_metadata["n_observations"]
    modeling_metadata["removed_duplicate_rows"] = cleaning_metadata["removed_duplicate_rows"]
    modeling_metadata["modeling_ready"] = cleaning_metadata["cleaning_status"] != "blocked_conflicting_labels"

    return DatasetBundle(
        name=name,
        fetch_key=fetch_key,
        X=cleaned_X,
        y=cleaned_y,
        original_y=cleaned_original_y,
        metadata=modeling_metadata,
        raw_X=X,
        raw_original_y=original_y,
        raw_metadata=raw_metadata,
        cleaning_metadata=cleaning_metadata,
    )


def load_dataset(name: str) -> DatasetBundle:
    if name not in REQUESTED_DATASETS:
        raise KeyError(f"Unknown requested dataset {name!r}. Expected one of {list(REQUESTED_DATASETS)}")
    raw = fetch_requested_raw_datasets()[name]
    return inspect_and_remap_dataset(name, REQUESTED_DATASETS[name], raw)


def load_all_datasets() -> dict[str, DatasetBundle]:
    raw = fetch_requested_raw_datasets()
    return {
        name: inspect_and_remap_dataset(name, REQUESTED_DATASETS[name], dataset)
        for name, dataset in raw.items()
    }


def dataset_summary_frame(bundles: dict[str, DatasetBundle], summary_kind: str = "modeling") -> pd.DataFrame:
    rows = []
    for bundle in bundles.values():
        meta = bundle.raw_metadata if summary_kind == "raw" else bundle.metadata
        modeling_ready = bundle.metadata.get("modeling_ready", True)
        blocked_modeling = summary_kind == "modeling" and not modeling_ready
        rows.append({
            "dataset": meta["dataset_name"],
            "fetch_key": meta["fetch_key"],
            "observations": None if blocked_modeling else meta["n_observations"],
            "input_features": meta["n_features"],
            "original_class_labels": json.dumps(meta["original_class_labels"]),
            "majority_class_label": meta["majority_class_label"],
            "minority_class_label": meta["minority_class_label"],
            "majority_class_count": None if blocked_modeling else meta["majority_class_count"],
            "minority_class_count": None if blocked_modeling else meta["minority_class_count"],
            "minority_class_percentage": None if blocked_modeling else meta["minority_class_percentage"],
            "imbalance_ratio": None if blocked_modeling else meta["imbalance_ratio"],
            "missing_values": meta["missing_values"],
            "duplicate_observations": None if blocked_modeling else meta["duplicate_observations"],
            "modeling_ready": modeling_ready,
            "cleaning_status": bundle.cleaning_metadata["cleaning_status"],
            "descriptive_statistics_json": "" if blocked_modeling else json.dumps(meta["descriptive_statistics"], sort_keys=True),
        })
    return pd.DataFrame(rows)


def dataset_cleaning_summary_frame(bundles: dict[str, DatasetBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles.values():
        meta = bundle.cleaning_metadata
        rows.append({
            "dataset": meta["dataset"],
            "raw_observations": meta["raw_observations"],
            "cleaned_observations": meta["cleaned_observations"],
            "removed_duplicate_rows": meta["removed_duplicate_rows"],
            "removed_majority_rows": meta["removed_majority_rows"],
            "removed_minority_rows": meta["removed_minority_rows"],
            "raw_majority_count": meta["raw_majority_count"],
            "raw_minority_count": meta["raw_minority_count"],
            "cleaned_majority_count": meta["cleaned_majority_count"],
            "cleaned_minority_count": meta["cleaned_minority_count"],
            "raw_imbalance_ratio": meta["raw_imbalance_ratio"],
            "cleaned_imbalance_ratio": meta["cleaned_imbalance_ratio"],
            "conflicting_feature_groups": meta["conflicting_feature_groups"],
            "conflicting_groups_json": meta["conflicting_groups_json"],
            "cleaning_status": meta["cleaning_status"],
        })
    return pd.DataFrame(rows)


def conflicting_feature_groups_frame(bundles: dict[str, DatasetBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles.values():
        conflicts = json.loads(bundle.cleaning_metadata["conflicting_groups_json"])
        for group_number, conflict in enumerate(conflicts, start=1):
            rows.append({
                "dataset": bundle.name,
                "group_number": group_number,
                "labels": json.dumps(conflict["labels"]),
                "row_indices": json.dumps(conflict["row_indices"]),
                "feature_values_json": json.dumps(conflict["feature_values"], sort_keys=True),
            })
    return pd.DataFrame(rows)


def label_mapping_frame(bundles: dict[str, DatasetBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles.values():
        meta = bundle.metadata
        rows.append({
            "dataset": meta["dataset_name"],
            "fetch_key": meta["fetch_key"],
            "original_majority_label": meta["majority_class_label"],
            "remapped_majority_label": NEGATIVE_LABEL,
            "original_minority_label": meta["minority_class_label"],
            "remapped_minority_label": POSITIVE_LABEL,
            "mapping_json": json.dumps(meta["label_mapping"], sort_keys=True),
        })
    return pd.DataFrame(rows)
