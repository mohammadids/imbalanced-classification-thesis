from __future__ import annotations

import importlib.metadata as metadata
import logging
import platform
import sys
from pathlib import Path

import pandas as pd

from .config import LOGS_DIR, TABLES_DIR
from .data_loader import (
    dataset_cleaning_summary_frame,
    conflicting_feature_groups_frame,
    dataset_summary_frame,
    label_mapping_frame,
    load_all_datasets,
)


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "matplotlib",
    "pytest",
    "jupyter",
)


def configure_logger(
    path: Path = LOGS_DIR / "phase1_validation.log",
    logger_name: str = "phase1_validation",
    mode: str = "w",
) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_software_versions(path: Path = LOGS_DIR / "software_versions.txt") -> dict[str, str]:
    versions = {}
    lines = []
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    lines.append(f"python=={platform.python_version()}")
    lines.append(f"platform=={platform.platform()}")
    lines.append(f"executable=={sys.executable}")
    for package in TRACKED_PACKAGES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed"
        versions[package] = version
        lines.append(f"{package}=={version}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return versions


def validate_bundles(bundles, require_modeling_ready: bool = False) -> None:
    for name, bundle in bundles.items():
        labels = sorted(bundle.y.unique().tolist())
        if labels != [0, 1]:
            raise ValueError(f"{name} was not remapped to [0, 1]; found {labels}")
        if int((bundle.y == 1).sum()) != bundle.metadata["minority_class_count"]:
            raise ValueError(f"{name} minority label was not remapped to 1")
        if int((bundle.y == 0).sum()) != bundle.metadata["majority_class_count"]:
            raise ValueError(f"{name} majority label was not remapped to 0")
        if require_modeling_ready and not bundle.metadata.get("modeling_ready", True):
            raise ValueError(f"{name} is not modeling-ready: {bundle.cleaning_metadata['cleaning_status']}")
        if not bundle.metadata.get("modeling_ready", True):
            continue
        combined = bundle.X.copy()
        combined["target"] = bundle.y
        if combined.duplicated().any():
            raise ValueError(f"{name} still contains exact feature-and-target duplicates")
        if bundle.cleaning_metadata["conflicting_feature_groups"] != 0:
            raise ValueError(f"{name} contains conflicting duplicate feature groups")


def generate_phase1_tables(bundles, tables_dir: Path = TABLES_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary_frame(bundles)
    mappings = label_mapping_frame(bundles)
    summary.to_csv(tables_dir / "dataset_summary.csv", index=False)
    mappings.to_csv(tables_dir / "class_label_mappings.csv", index=False)
    return summary, mappings


def generate_phase1_5_tables(bundles, tables_dir: Path = TABLES_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_summary = dataset_summary_frame(bundles, summary_kind="raw")
    modeling_summary = dataset_summary_frame(bundles, summary_kind="modeling")
    cleaning_summary = dataset_cleaning_summary_frame(bundles)
    conflicts = conflicting_feature_groups_frame(bundles)
    raw_summary.to_csv(tables_dir / "dataset_summary_raw.csv", index=False)
    modeling_summary.to_csv(tables_dir / "dataset_summary_modeling.csv", index=False)
    cleaning_summary.to_csv(tables_dir / "dataset_cleaning_summary.csv", index=False)
    conflicts.to_csv(tables_dir / "conflicting_feature_groups.csv", index=False)
    return raw_summary, modeling_summary, cleaning_summary


def generate_phase1_6_tables(bundles, tables_dir: Path = TABLES_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_summary = dataset_summary_frame(bundles, summary_kind="raw")
    modeling_summary = dataset_summary_frame(bundles, summary_kind="modeling")
    cleaning_summary = dataset_cleaning_summary_frame(bundles)
    mappings = label_mapping_frame(bundles)
    raw_summary.to_csv(tables_dir / "dataset_summary_raw_phase1_6.csv", index=False)
    modeling_summary.to_csv(tables_dir / "dataset_summary_modeling_phase1_6.csv", index=False)
    cleaning_summary.to_csv(tables_dir / "dataset_cleaning_summary_phase1_6.csv", index=False)
    mappings.to_csv(tables_dir / "class_label_mappings_phase1_6.csv", index=False)
    return raw_summary, modeling_summary, cleaning_summary, mappings


def load_validate_and_write_tables(logger: logging.Logger | None = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("loading requested datasets")
    bundles = load_all_datasets()
    logger.info("validating binary labels and remapping")
    validate_bundles(bundles)
    logger.info("writing dataset summary and label mapping tables")
    summary, mappings = generate_phase1_tables(bundles)
    return bundles, summary, mappings


def load_validate_and_write_phase1_5_tables(logger: logging.Logger | None = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("loading requested datasets with duplicate-cleaning procedure")
    bundles = load_all_datasets()
    logger.info("validating cleaned modeling datasets")
    validate_bundles(bundles)
    logger.info("writing Phase 1.5 raw/modeling/cleaning tables")
    raw_summary, modeling_summary, cleaning_summary = generate_phase1_5_tables(bundles)
    return bundles, raw_summary, modeling_summary, cleaning_summary


def load_validate_and_write_phase1_6_tables(logger: logging.Logger | None = None):
    logger = logger or logging.getLogger(__name__)
    logger.info("loading final Phase 1.6 requested datasets")
    bundles = load_all_datasets()
    logger.info("validating final datasets are binary, remapped, cleaned, and modeling-ready")
    validate_bundles(bundles, require_modeling_ready=True)
    logger.info("writing Phase 1.6 raw/modeling/cleaning/mapping tables")
    raw_summary, modeling_summary, cleaning_summary, mappings = generate_phase1_6_tables(bundles)
    return bundles, raw_summary, modeling_summary, cleaning_summary, mappings
