# tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True)
def override_test_paths(monkeypatch):
    """
    Point the app at our tiny fixtures instead of the full dataset.
    """
    monkeypatch.setenv(
        "INDEX_PATH",
        "tests/fixtures/data_small/ivf_flat_small.index",
    )
    monkeypatch.setenv(
        "META_PATH",
        "tests/fixtures/data_small/id2meta_small.json",
    )
