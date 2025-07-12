"""
app/main.py
~~~~~~~~~~~

• Serves a minimal cross-modal search API backed by a FAISS index.
• Automatically instruments Prometheus metrics at /metrics
• Works out-of-the-box with either the full dataset or the 1 000-item
  fixture set in tests/fixtures/data_small/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from prometheus_fastapi_instrumentator import Instrumentator


# 1. Locate artefacts – fall back to the tiny fixtures if env-vars are absent
REPO_ROOT = Path(__file__).resolve().parent.parent         # …/step11
DEFAULT_INDEX = REPO_ROOT / "tests" / "fixtures" / "data_small" / "ivf_flat_small.index"
DEFAULT_META  = REPO_ROOT / "tests" / "fixtures" / "data_small" / "id2meta_small.json"

INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DEFAULT_INDEX)).resolve()
META_PATH  = Path(os.getenv("META_PATH",       DEFAULT_META )).resolve()

if not INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
if not META_PATH.exists():
    raise RuntimeError(f"Metadata json not found: {META_PATH}")

# 2. Load artefacts once at startup
INDEX: faiss.Index     # will be populated in _startup_event()
METADATA: list[dict]   # ditto


def _load_artefacts() -> None:
    global INDEX, METADATA                              # pylint: disable=global-statement
    INDEX = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        METADATA = json.load(f)            # list[{"id": …, "caption": …, …}]
    if len(METADATA) != INDEX.ntotal:
        raise RuntimeError(
            f"Mismatched sizes – index={INDEX.ntotal}  meta={len(METADATA)}"
        )


# 3. Pydantic schemas for requests / responses
class SearchRequest(BaseModel):
    query_vec: conlist(float, min_items=512, max_items=512) = Field(
        ..., description="512-dim unit-norm CLIP embedding"
    )
    k: int = Field(3, ge=1, le=50, description="how many nearest neighbours")


class SearchResult(BaseModel):
    id: int
    caption: str
    score: float


# 4. Core search helper
def _search(query: list[float], k: int) -> List[SearchResult]:
    # FAISS expects (n, dim) float32 row-major
    q = np.asarray(query, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)

    scores, idxs = INDEX.search(q, k)          # shape (1, k)
    results: list[SearchResult] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        meta = METADATA[int(idx)]
        results.append(
            SearchResult(id=meta["id"], caption=meta["caption"], score=float(score))
        )
    return results


# 5. FastAPI instance + Prometheus instrumentation
app = FastAPI(title="Capstone Step-11 Search API")
Instrumentator().instrument(app).expose(app)   # adds /metrics


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=List[SearchResult])
def search(payload: SearchRequest):
    if len(payload.query_vec) != INDEX.d:
        raise HTTPException(
            status_code=400,
            detail=f"Expected vector of length {INDEX.d}",
        )
    return _search(payload.query_vec, payload.k)


# 6. FastAPI lifecycle hook – load everything exactly once
@app.on_event("startup")
def _startup_event() -> None:   # noqa: D401  (FastAPI signature requirement)
    _load_artefacts()
