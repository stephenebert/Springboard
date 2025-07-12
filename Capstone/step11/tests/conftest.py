# tests/conftest.py
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def override_fixture_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Point the app at the 1 000-item fixture set that lives under
    tests/fixtures/data_small/.
    """
    root = Path(__file__).resolve().parents[1]          # repo-root / tests
    fixtures = root / "fixtures" / "data_small"

    monkeypatch.setenv(
        "FAISS_INDEX_PATH",                # matches main.py
        str(fixtures / "ivf_flat_small.index"),
    )
    monkeypatch.setenv(
        "META_PATH",
        str(fixtures / "id2meta_small.json"),
    )
