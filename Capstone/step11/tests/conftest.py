# tests/conftest.py

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# 1) Point the app at our tiny fixtures
DATA_DIR = Path(__file__).parent / "fixtures" / "data_small"
os.environ["FAISS_INDEX_PATH"] = str(DATA_DIR / "ivf_flat_small.index")
os.environ["META_PATH"]       = str(DATA_DIR / "id2meta_small.json")

# 2) Now import the app (it will pick up the two env vars above)
from app.main import app

# 3) Provide a TestClient for all tests
@pytest.fixture(scope="session")
def client():
    """
    A FastAPI TestClient that runs the app in‐process.
    """
    return TestClient(app)
