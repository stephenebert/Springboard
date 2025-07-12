# app/main.py
import os
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1) Paths & defaults
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # → Capstone/step11
FIXTURES = ROOT / "tests" / "fixtures" / "data_small"

EMBS_PATH = Path(os.getenv("EMBS_PATH", FIXTURES / "img_embs_small.npy"))
META_PATH = Path(os.getenv("META_PATH", FIXTURES / "id2meta_small.json"))

# ---------------------------------------------------------------------------
# 2) Globals
# ---------------------------------------------------------------------------
INDEX: faiss.Index = None
METADATA: Dict[int, Dict] = {}

# ---------------------------------------------------------------------------
# 3) Pydantic models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query_vec: List[float]
    k: int


class SearchResult(BaseModel):
    id: int
    caption: str
    url: str
    score: float


# ---------------------------------------------------------------------------
# 4) FastAPI
# ---------------------------------------------------------------------------
app = FastAPI()


@app.on_event("startup")
def load_artifacts() -> None:
    """Create a FAISS index from the tiny fixture embeddings & load metadata."""
    global INDEX, METADATA

    # ---- load & normalise embeddings -------------------------------------------------
    if not EMBS_PATH.is_file():
        raise RuntimeError(f"Embeddings not found: {EMBS_PATH}")
    embs = np.load(EMBS_PATH).astype("float32")
    faiss.normalize_L2(embs)                       # cosine → inner-product

    # ---- build an in-memory index ----------------------------------------------------
    INDEX = faiss.IndexFlatIP(embs.shape[1])       # inner-product (=cosine)
    INDEX.add(embs)

    # ---- metadata -------------------------------------------------------------------
    if not META_PATH.is_file():
        raise RuntimeError(f"Metadata not found: {META_PATH}")
    with META_PATH.open() as f:
        raw = json.load(f)
    METADATA = {int(k): v for k, v in raw.items()}


@app.get("/health")
def health():
    """Simple liveness check used by the tests."""
    if INDEX is None:
        raise HTTPException(503, "Index not loaded")
    return {
        "status": "ok",
        "index_dim": INDEX.d,
        "index_size": INDEX.ntotal,
    }


def _search(vec: List[float], k: int) -> List[SearchResult]:
    q = np.asarray(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)

    distances, indices = INDEX.search(q, k)
    results: List[SearchResult] = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx not in METADATA:
            continue
        meta = METADATA[idx]
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
    if INDEX is None:
        raise HTTPException(503, "Index not loaded")
    if len(payload.query_vec) != INDEX.d:
        raise HTTPException(400, f"Expected vector of length {INDEX.d}")
    return _search(payload.query_vec, payload.k)
