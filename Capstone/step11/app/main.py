# app/main.py

import os
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# 1) Paths & defaults
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = Path(
    os.getenv(
        "FAISS_INDEX_PATH",
        ROOT / "tests" / "fixtures" / "data_small" / "ivf_flat_small.index",
    )
)
META_PATH = Path(
    os.getenv(
        "META_PATH",
        ROOT / "tests" / "fixtures" / "data_small" / "id2meta_small.json",
    )
)

# -----------------------------------------------------------------------------
# 2) Globals to hold your loaded artifacts
# -----------------------------------------------------------------------------
INDEX: faiss.Index = None
METADATA: Dict[str, Dict] = {}

# -----------------------------------------------------------------------------
# 3) Pydantic models
# -----------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query_vec: List[float]
    k: int

class SearchResult(BaseModel):
    id: int
    caption: str
    url: str
    score: float

# -----------------------------------------------------------------------------
# 4) FastAPI setup
# -----------------------------------------------------------------------------
app = FastAPI()


@app.on_event("startup")
def load_artifacts():
    """
    Read the FAISS index and the (tiny) metadata dictionary into globals.
    """
    global INDEX, METADATA

    if not INDEX_PATH.is_file():
        raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
    if not META_PATH.is_file():
        raise RuntimeError(f"Metadata file not found: {META_PATH}")

    INDEX = faiss.read_index(str(INDEX_PATH))

    # ── CHANGED ── convert JSON keys (strings) → int  ────────────────
    with open(META_PATH, "r") as f:
        raw = json.load(f)
    METADATA = {int(k): v for k, v in raw.items()}
    # ────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """
    Minimal liveness + a tiny bit of index info so the tests can assert on it.
    """
    if INDEX is None:
        raise HTTPException(503, "Index not loaded")
    return {
        "status": "ok",
        "index_dim": INDEX.d,
        "index_size": INDEX.ntotal,
    }


def _search(vec: List[float], k: int) -> List[SearchResult]:
    """
    Helper that does the FAISS search and maps ids → metadata.
    """
    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    distances, indices = INDEX.search(q, k)

    results: List[SearchResult] = []
    for dist, idx in zip(distances[0], indices[0]):

        # ── CHANGED ── ignore padding ids (-1) and look up by int ────
        if idx == -1 or idx not in METADATA:
            continue
        meta = METADATA[idx]
        # ────────────────────────────────────────────────────────────

        results.append(
            SearchResult(
                id=idx,
                caption=meta["caption"],
                url=meta["url"],
                score=float(dist),
            )
        )
    return results


@app.post("/search", response_model=List[SearchResult])
def search(payload: SearchRequest):
    """
    Main search endpoint: validates the vector length and calls _search().
    """
    if INDEX is None:
        raise HTTPException(503, "Index not loaded")
    if len(payload.query_vec) != INDEX.d:
        raise HTTPException(400, f"Expected vector of length {INDEX.d}")

    return _search(payload.query_vec, payload.k)
