# app/main.py

import os
import json
from pathlib import Path
from typing import List

import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, Field
from starlette.middleware import Middleware
from prometheus_fastapi_instrumentator import InstrumentatorMiddleware, Instrumentator

# 1) Figure out repo‐root relative defaults

# this file lives in <repo>/app/main.py, so repo root is two levels up
REPO_ROOT = Path(__file__).parent.parent.resolve()

# default index path: <repo>/data/ivf_flat_1024.index
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "ivf_flat_1024.index"
# default meta path:    <repo>/data/id2meta.json
DEFAULT_META_PATH  = REPO_ROOT / "data" / "id2meta.json"

# allow overrides via environment
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(DEFAULT_INDEX_PATH))).resolve()
META_PATH  = Path(os.getenv("META_PATH",       str(DEFAULT_META_PATH))).resolve()

# sanity‐check
if not INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
if not META_PATH.exists():
    raise RuntimeError(f"Metadata file not found: {META_PATH}")


# 2) Declare Pydantic models

class SearchRequest(BaseModel):
    query_vec: conlist(float, min_items=None) = Field(
        ..., description="Normalized float32 vector of length d"
    )
    k: int = Field(..., gt=0, description="Number of nearest neighbours to return")


class SearchResult(BaseModel):
    id: int
    caption: str
    score: float


# 3) Prepare app (with Prometheus middleware)

middleware = [
    Middleware(InstrumentatorMiddleware)  # exposes /metrics
]
app = FastAPI(middleware=middleware)

# attach prometheus instrumentator
Instrumentator().instrument(app).expose(app)


# 4) In‐memory artifacts (populated on startup)

INDEX: faiss.Index = None
METADATA: List[dict] = []


def _load_artifacts():
    global INDEX, METADATA

    # 1) load metadata JSON (list of dicts with "id" and "caption" keys)
    with open(META_PATH, "r", encoding="utf-8") as f:
        METADATA = json.load(f)

    # 2) load and normalize FAISS index
    INDEX = faiss.read_index(str(INDEX_PATH))
    # note: assume index expects normalized vectors
    faiss.normalize_L2(INDEX.reconstruct_n(0, 1))  # warm up
    # store the embedding dim for easy checking
    INDEX.d = INDEX.d  # faiss keeps .d attribute


def _search(query_vec: List[float], k: int) -> List[SearchResult]:
    # run FAISS search
    q = query_vec.copy()
    faiss.normalize_L2(q)
    D, I = INDEX.search(  # distances, indices
        faiss.VectorFloat(q),  # you may need to convert to numpy array
        k
    )
    results: List[SearchResult] = []
    for dist, idx in zip(D[0], I[0]):
        meta = METADATA[idx]
        results.append(SearchResult(
            id=meta["id"],
            caption=meta["caption"],
            score=float(dist)
        ))
    return results

# 5) API endpoints

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_dim": INDEX.d,
        "index_size": INDEX.ntotal,
        "nprobe": getattr(INDEX, "nprobe", None),
    }


@app.post("/search", response_model=List[SearchResult])
def search(payload: SearchRequest):
    if len(payload.query_vec) != INDEX.d:
        raise HTTPException(
            status_code=400,
            detail=f"Expected vector of length {INDEX.d}"
        )
    return _search(payload.query_vec, payload.k)


# 6) Lifespan (startup) handler

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

@app.router.lifespan  # new style
async def _lifespan(app: FastAPI, receive: Receive, send: Send):
    # load index + metadata before handling any requests
    _load_artifacts()
    # then yield control back to the server
    yield

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

