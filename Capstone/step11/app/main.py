#!/usr/bin/env python
"""
FastAPI retrieval service backed by the Step 8 IVF-Flat index.

Environment variables (with sensible defaults)
-----------------------------------------------
FAISS_INDEX_PATH   Path to the .index file     [/data/ivf_flat_1024.index]
META_PATH          Path to id2meta.json        [/data/id2meta.json]
NPROBE             FAISS nprobe at runtime     [16]

Endpoints
-----------------------------------------------
GET  /health                       - liveness & config
POST /search {query_vec,k} --> top-K - ANN search (cos-sim)
GET  /metrics                      - Prometheus scraper endpoint
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Prometheus instrumentation 
from prometheus_fastapi_instrumentator import Instrumentator


# Startup: load FAISS index + metadata once, pin in RAM
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "/data/ivf_flat_1024.index"))
META_PATH  = Path(os.getenv("META_PATH",       "/data/id2meta.json"))
NPROBE     = int(os.getenv("NPROBE", "16"))

if not INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
if not META_PATH.exists():
    raise RuntimeError(f"Metadata JSON not found: {META_PATH}")

faiss_index = faiss.read_index(str(INDEX_PATH))
faiss_index.nprobe = NPROBE
DIM = faiss_index.d

with META_PATH.open("r", encoding="utf-8") as f:
    ID2META: list[dict] = json.load(f)

# FastAPI wiring
app = FastAPI(title="Step 8 Retrieval Service", version="0.1.0")

# expose /metrics and add request/latency counters
Instrumentator().instrument(app).expose(app)         


class SearchRequest(BaseModel):
    query_vec: List[float] = Field(
        ...,
        description="512-D CLIP embedding (image or text)",
        min_items=DIM,
        max_items=DIM,
    )
    k: int = Field(5, ge=1, le=50, description="Number of nearest neighbours to return")


@app.get("/health", summary="Liveness probe & index stats")
def health():
    return {
        "status": "ok",
        "index_dim": DIM,
        "nprobe": faiss_index.nprobe,
        "index_size": faiss_index.ntotal,
    }


@app.post("/search", summary="ANN search (cosine similarity)")
def search(req: SearchRequest):
    vec = np.asarray(req.query_vec, dtype="float32")
    if vec.shape[0] != DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension {vec.shape[0]} ≠ index dim {DIM}",
        )

    vec = vec[None, :]
    faiss.normalize_L2(vec)            # cosine similarity via inner-product
    D, I = faiss_index.search(vec, req.k)

    results = [
    {
        "id": int(idx),
        "image_path": ID2META[idx]["image_path"],  
        "caption":    ID2META[idx]["caption"],
        "score": float(score),
    }
    for idx, score in zip(I[0], D[0])
]

    return {"k": req.k, "results": results}
