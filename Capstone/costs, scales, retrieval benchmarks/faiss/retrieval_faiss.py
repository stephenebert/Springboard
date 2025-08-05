"""
retrieval_faiss.py
──────────────────
Minimal helper that loads a FAISS index + the COCO-val captions and exposes
a single function:

    search(vec: np.ndarray, top_k: int = 3) -> list[str]

Returned list is *sorted* by distance (nearest first).

Works with any FAISS index / embedding dimensionality as long as
the index’s ntotal == len(captions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss                    # pip install faiss-cpu
import numpy as np


# ── CONFIG ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

INDEX_PATH  = ROOT / "hnsw_val2017_imagebind.faiss"            # <- new index
CAPT_PATH   = ROOT / "data/coco/annotations/captions_val2017.json"

# if you keep multiple indices around, change only INDEX_PATH above

# ── LOAD CAPTIONS ─────────────────────────────────────────────────────
with CAPT_PATH.open() as f:
    coco = json.load(f)
CAPTIONS: List[str] = [ann["caption"] for ann in coco["annotations"]]
print(f"• loaded {len(CAPTIONS):,} captions")

# ── LOAD FAISS INDEX ─────────────────────────────────────────────────
print("• loading FAISS index …")
INDEX = faiss.read_index(str(INDEX_PATH))
DIM   = INDEX.d                       # embedding dimensionality
print(f"  ↳ dim={DIM} | engine={type(INDEX).__name__} | efSearch={getattr(INDEX, 'efSearch', 'n/a')}")

# sanity-check
assert INDEX.ntotal == len(CAPTIONS), (
    f"Index holds {INDEX.ntotal:,} vectors but {len(CAPTIONS):,} captions loaded – "
    "make sure you embedded the *same* caption set used for the index!"
)
# ──────────────────────────────────────────────────────────────────────


def _np_vectorize(x: np.ndarray | list[float]) -> np.ndarray:
    """Convert a single vector to shape (1, dim) float32 numpy array."""
    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x[None, :]                 # (dim,) -> (1, dim)
    if x.shape[1] != DIM:
        raise ValueError(f"Vector dim {x.shape[1]} ≠ index dim {DIM}")
    return x



def search(vec: np.ndarray, top_k: int = 3) -> list[tuple[float, str]]:
    """
    Return [(distance, caption), …] sorted nearest-to-furthest.
    """
    # ── sanity ─────────────────────────────────────────────────────────────
    if vec.ndim == 1:
        vec = vec[None, :]                     # (1, dim)
    assert vec.shape[1] == INDEX.d, "vector dim mismatch"

    D, I = INDEX.search(vec.astype("float32"), top_k)   # D, I → shape (1, k)

    hits: list[tuple[float, str]] = [
        (float(D[0, j]), CAPTIONS[int(I[0, j])])        # (score, caption)
        for j in range(I.shape[1])
        if int(I[0, j]) != -1                           # -1 = “no neighbour”
    ]
    return hits


# ── REPL test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # quick smoke-test with the 1st vector in the index
    xb, _ = INDEX.sa_encode() if hasattr(INDEX, "sa_encode") else (None, None)
    if xb is not None:
        print("[debug] index stores inverted‐lists – can’t extract raw vectors")
    else:
        xb = INDEX.reconstruct_n(0, 1)         # (1, dim)
        print("nearest to caption[0]:", search(xb[0], 5))

