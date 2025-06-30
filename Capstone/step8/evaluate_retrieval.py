"""
evaluate_retrieval.py

Measure indexing time, memory, and per-query latency
on the full image-embedding store, with FAISS.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import h5py
import psutil

try:
    import faiss
    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False


def load_embeddings(path: Path) -> np.ndarray:
    """Load image embeddings from .h5 or fallback to img_embs_full.npy."""
    if path.suffix.lower() == ".h5" and path.exists():
        with h5py.File(str(path), "r") as hf:
            return hf["image_embeddings"][:]
    else:
        npy = path.with_name("img_embs_full.npy")
        if npy.exists():
            return np.load(str(npy))
    raise FileNotFoundError(f"Embeddings not found at {path} or fallback .npy")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate image→image retrieval speed & memory"
    )
    p.add_argument(
        "--h5",
        type=Path,
        default=Path("experiments/full/embeddings_full.h5"),
        help="Path to embeddings_full.h5 (defaults to experiments/full/embeddings_full.h5)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Recall@K (number of neighbors to search)",
    )
    p.add_argument(
        "--nq",
        type=int,
        default=1000,
        help="Number of random queries to time",
    )
    args = p.parse_args()

    # Load
    print(f"\n Loading embeddings from {args.h5}")
    emb = load_embeddings(args.h5).astype("float32")
    N, D = emb.shape
    print(f"Loaded   {N:,} vectors  (dim={D})")

    # Build index
    if _HAVE_FAISS:
        print("\n Building FAISS IndexFlatIP...")
        index = faiss.IndexFlatIP(D)
        t0 = time.time()
        batch_size = 100_000
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            index.add(emb[start:end])
        print(f"Index built in {time.time()-t0:.1f}s")
    else:
        print("\n  FAISS not installed—falling back to NumPy brute force")
        index = None

    # Report memory
    rss_mb = psutil.Process().memory_info().rss / 1024**2
    print(f"\n RSS after indexing: {rss_mb:.1f} MB")

    # Sample queries
    rng = np.random.default_rng(0)
    selects = rng.choice(N, size=args.nq, replace=False)
    queries = emb[selects]


    print("\n Warm-up...")
    if index:
        _ = index.search(queries[:10], args.k)
    else:
        sims = queries[:10] @ emb.T
        _ = np.argpartition(-sims, args.k, axis=1)[:, : args.k]

    print(f"\n Timing {args.nq} queries at K={args.k}...")
    t0 = time.time()
    if index:
        Dists, Ids = index.search(queries, args.k)
    else:
        sims = queries @ emb.T
        _ = np.argpartition(-sims, args.k, axis=1)[:, : args.k]
    avg_ms = (time.time() - t0) * 1000 / args.nq
    print(f" Avg latency: {avg_ms:.2f} ms per query\n")


if __name__ == "__main__":
    main()
"""
Loading embeddings from experiments\full\embeddings_full.h5
Loaded   850,668 vectors  (dim=512)

  FAISS not installed falling back to NumPy brute force

 RSS after indexing: 1702.7 MB

 Warm-up...

 Timing 1000 queries at K=10...
 Avg latency: 37.17 ms per query
"""