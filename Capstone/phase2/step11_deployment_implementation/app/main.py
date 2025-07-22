#!/usr/bin/env python
"""
FastAPI retrieval service backed by the IVF-Flat index.

Env vars
--------
FAISS_INDEX_PATH   Path to the .index file     [/data/ivf_flat_1024.index]
META_PATH          Path to id2meta.json        [/data/id2meta.json]
NPROBE             FAISS nprobe at runtime     [16]

Endpoints
---------
GET  /health                       — liveness & config
POST /search {caption?,query_vec?,k} → top-K
GET  /metrics                      — Prometheus metrics
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# CONFIG
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "/data/ivf_flat_1024.index"))
META_PATH  = Path(os.getenv("META_PATH",       "/data/id2meta.json"))
NPROBE     = int(os.getenv("NPROBE", "16"))

if not INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
if not META_PATH.exists():
    raise RuntimeError(f"Metadata JSON not found: {META_PATH}")

# LOAD INDEX & METADATA
faiss_index = faiss.read_index(str(INDEX_PATH))
faiss_index.nprobe = NPROBE
DIM = faiss_index.d

with META_PATH.open("r", encoding="utf-8") as f:
    ID2META: List[dict] = json.load(f)

# FASTAPI APP & METRICS
app = FastAPI(title="Image↔Text Retrieval API", version="0.1.0")
Instrumentator().instrument(app).expose(app)


# Pydantic models
class SearchRequest(BaseModel):
    caption:   Optional[str]          = Field(None, description="Text prompt to encode")
    query_vec: Optional[List[float]]  = Field(
        None,
        description=f"Precomputed {DIM}-dim embedding",
        min_items=DIM,
        max_items=DIM,
    )
    k:         int                    = Field(5, ge=1, le=100, description="Top-K results")


class Neighbor(BaseModel):
    id:         int
    caption:    str
    image_path: str
    score:      float


class SearchResponse(BaseModel):
    k:       int
    results: List[Neighbor]


# Placeholder text encoder (swap in your CLIP encoder)
def encode_text_placeholder(text: str) -> List[float]:
    """
    Deterministic random vector per string (placeholder for real CLIP encode).
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.standard_normal(DIM, dtype="float32")
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


# HEALTH ENDPOINT
@app.get("/health", summary="Liveness probe & index stats")
def health():
    return {
        "status":     "ok",
        "index_dim":  DIM,
        "nprobe":     faiss_index.nprobe,
        "index_size": faiss_index.ntotal,
    }


# SEARCH ENDPOINT
@app.post("/search", response_model=SearchResponse, summary="ANN search")
def search(req: SearchRequest):
    # 1) Determine which vector to use
    if req.caption:
        vec = encode_text_placeholder(req.caption)
    elif req.query_vec is not None:
        vec = req.query_vec
        if len(vec) != DIM:
            raise HTTPException(
                status_code=422,
                detail=f"query_vec must have length {DIM}, got {len(vec)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either `caption` or `query_vec` in request body"
        )

    # 2) Run FAISS
    xq = np.array(vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(xq)           # use inner-product as cosine sim
    D, I = faiss_index.search(xq, req.k)

    # 3) Build results
    results: List[Neighbor] = []
    for score, idx in zip(D[0], I[0]):
        meta = ID2META[idx]
        results.append(
            Neighbor(
                id=int(idx),
                caption=meta["caption"],
                image_path=meta["image_path"],
                score=float(score),
            )
        )

    return SearchResponse(k=req.k, results=results)
