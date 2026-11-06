import pytest

from src.data_loader import load_all_datasets


@pytest.fixture(scope="session")
def bundles():
    return load_all_datasets()

