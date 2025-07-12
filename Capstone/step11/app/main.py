# app/main.py
import os
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------#
# 1) Paths & defaults                                                          #
# -----------------------------------------------------------------------------#
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

# -----------------------------------------------------------------------------#
# 2) Globals to hold loaded artefacts                                          #
# -----------------------------------------------------------------------------#
INDEX: faiss.Index | None = None            # type: ignore[assignment]
METADATA: Dict[str, Dict] = {}

# -----------------------------------------------------------------------------#
# 3) Pydantic models                                                           #
# -----------------------------------------------------------------------------#
class SearchRequest(BaseModel):
    query_vec: List[float]
    k: int

class SearchResult(BaseModel):
    id: int
    caption: str
    url: str
    score: float

# -----------------------------------------------------------------------------#
# 4) FastAPI                                                                   #
# -----------------------------------------------------------------------------#
app = FastAPI()

@app.on_event("startup")
def _load_artifacts() -> None:
    """Read the tiny FAISS index & metadata at start-up."""
    global INDEX, METADATA

    if not INDEX_PATH.is_file():
        raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
    if not META_PATH.is_file():
        raise RuntimeError(f"Metadata file not found: {META_PATH}")

    INDEX = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        METADATA = json.load(f)

@app.get("/health")
def health() -> Dict:
    """Simple readiness probe used by the tests."""
    return {
        "status": "ok",
        "index_dim": INDEX.d,            # e.g. 512
        "index_size": INDEX.ntotal,      # number of vectors
    }

# -----------------------------------------------------------------------------#
# 5) Internal helpers                                                          #
# -----------------------------------------------------------------------------#
def _search(vec: List[float], k: int) -> List[SearchResult]:
    """Low-level FAISS search → List[SearchResult]."""
    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    k = max(1, min(k, INDEX.ntotal))   # clamp 1 ≤ k ≤ N

    distances, indices = INDEX.search(q, k)
    results: List[SearchResult] = []

    for dist, idx in zip(distances[0], indices[0]):
        # Metadata may be missing for a given id – supply safe defaults.
        meta = METADATA.get(str(int(idx)), {})
        results.append(
            SearchResult(
                id=int(idx),
                caption=meta.get("caption", ""),
                url=meta.get("url", ""),
                score=float(dist),
            )
        )
    return results

# -----------------------------------------------------------------------------#
# 6) Public endpoint                                                           #
# -----------------------------------------------------------------------------#
@app.post("/search", response_model=List[SearchResult])
def search(payload: SearchRequest):
    if INDEX is None:
        raise HTTPException(503, "Index not loaded")

    if len(payload.query_vec) != INDEX.d:
        raise HTTPException(
            status_code=400,
            detail=f"Expected vector of length {INDEX.d}, got {len(payload.query_vec)}",
        )

    return _search(payload.query_vec, payload.k)

