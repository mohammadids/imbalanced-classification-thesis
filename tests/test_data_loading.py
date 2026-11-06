from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import REQUESTED_DATASETS
from src.data_loader import fetch_requested_raw_datasets, inspect_and_remap_dataset


def test_all_three_datasets_can_be_loaded(bundles):
    assert set(bundles) == set(REQUESTED_DATASETS)
    assert set(bundles) == {"satimage", "ozone_level", "abalone_19"}
    for name, bundle in bundles.items():
        assert bundle.fetch_key == REQUESTED_DATASETS[name]
        assert bundle.X.shape[0] == bundle.y.shape[0]
        assert bundle.X.shape[0] > 0
        assert bundle.X.shape[1] > 0


def test_each_dataset_is_binary(bundles):
    for bundle in bundles.values():
        assert len(set(bundle.original_y)) == 2
        assert sorted(bundle.y.unique().tolist()) == [0, 1]


def test_all_final_datasets_are_modeling_ready(bundles):
    for bundle in bundles.values():
        assert bundle.metadata["modeling_ready"] is True
        assert bundle.cleaning_metadata["cleaning_status"] in {"unchanged", "cleaned"}


def test_no_exact_feature_target_duplicates_remain_in_modeling_ready_data(bundles):
    for bundle in bundles.values():
        if not bundle.metadata["modeling_ready"]:
            continue
        combined = bundle.X.copy()
        combined["target"] = bundle.y
        assert not combined.duplicated().any()


def test_conflicting_labels_are_not_silently_modified():
    raw = SimpleNamespace(
        data=np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        target=np.array([-1, 1, -1, -1]),
    )
    bundle = inspect_and_remap_dataset("conflict_demo", "conflict_demo", raw)
    assert bundle.cleaning_metadata["cleaning_status"] == "blocked_conflicting_labels"
    assert bundle.cleaning_metadata["conflicting_feature_groups"] == 1
    assert bundle.metadata["modeling_ready"] is False
    assert bundle.X.shape[0] == bundle.raw_X.shape[0]


def test_only_requested_datasets_are_fetched(monkeypatch):
    observed = {}

    def fake_fetch_datasets(filter_data, download_if_missing, verbose):
        observed["filter_data"] = filter_data
        return {
            "satimage": SimpleNamespace(data=pd.DataFrame([[0], [1]]), target=np.array([-1, 1])),
            "ozone_level": SimpleNamespace(data=pd.DataFrame([[0], [1]]), target=np.array([-1, 1])),
            "abalone_19": SimpleNamespace(data=pd.DataFrame([[0], [1]]), target=np.array([-1, 1])),
        }

    monkeypatch.setattr("src.data_loader.fetch_datasets", fake_fetch_datasets)
    fetched = fetch_requested_raw_datasets()
    assert observed["filter_data"] == tuple(REQUESTED_DATASETS.values())
    assert set(fetched) == {"satimage", "ozone_level", "abalone_19"}
