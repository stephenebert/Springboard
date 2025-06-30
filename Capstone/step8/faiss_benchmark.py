"""
faiss_benchmark.py

Compare exact (IndexFlatIP) vs approximate (IndexIVFFlat) retrieval speed & recall.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import h5py
import psutil
import faiss

def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".h5" and path.exists():
        with h5py.File(str(path), "r") as hf:
            return hf["image_embeddings"][:]
    else:
        npy = path.with_name("img_embs_full.npy")
        if npy.exists():
            return np.load(str(npy))
    raise FileNotFoundError(f"Embeddings not found at {path} or fallback .npy")

def memory_mb():
    return psutil.Process().memory_info().rss / 1024**2

def benchmark_index(name, index, embeddings, queries, k):
    print(f"\n--- {name} ---")
    t0 = time.time()
    if name.startswith("IVF"):
        # IVF needs training first
        ntrain = min(100_000, embeddings.shape[0])
        index.train(embeddings[:ntrain])
    index.add(embeddings)
    print(f"Indexing time: {time.time()-t0:.1f}s | RSS: {memory_mb():.1f} MB")

    # warm-up
    _ = index.search(queries[:10], k)

    # timed query
    t0 = time.time()
    D, I = index.search(queries, k)
    latency = (time.time()-t0) * 1000 / len(queries)
    print(f"Avg latency (@{k}): {latency:.2f} ms/query")
    return I

def compute_recall(knn_idx, ground_truth_idxs, k):
    # ground_truth_idxs is an array of length N where gt[i]=i
    hits = (knn_idx == ground_truth_idxs[:,None]).any(axis=1)
    return 100 * hits.mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5", type=Path,
        default=Path("experiments/full/embeddings_full.h5")
    )
    p.add_argument("--k",   type=int, default=10)
    p.add_argument("--nq",  type=int, default=2000)
    p.add_argument("--nlist", type=int, default=1024,
                   help="Number of IVF cells")
    args = p.parse_args()

    print(f"\nLoading embeddings from {args.h5}")
    emb = load_embeddings(args.h5).astype("float32")
    N, D = emb.shape
    print(f"Loaded {N:,} x {D}")

    # sample queries
    rng = np.random.default_rng(0)
    qidx = rng.choice(N, size=args.nq, replace=False)
    queries = emb[qidx]

    # exact brute-force ground truth
    print("\nComputing ground-truth via NumPy brute force...")
    sims = queries @ emb.T
    gt = np.argpartition(-sims, args.k-1, axis=1)[:, :args.k]

    # Exact Flat IP
    flat = faiss.IndexFlatIP(D)
    idx_flat = benchmark_index("Flat (exact)", flat, emb, queries, args.k)

    # IVF-FLAT
    quantizer = faiss.IndexFlatIP(D)
    ivf = faiss.IndexIVFFlat(quantizer, D, args.nlist, faiss.METRIC_INNER_PRODUCT)
    idx_ivf = benchmark_index(f"IVF-Flat nlist={args.nlist}", ivf, emb, queries, args.k)

    # recall
    r_flat = compute_recall(idx_flat, np.arange(args.nq), args.k)
    r_ivf  = compute_recall(idx_ivf,  np.arange(args.nq), args.k)
    print(f"\nRecall@{args.k}: Flat={r_flat:.2f}%, IVF-Flat={r_ivf:.2f}%")

if __name__ == "__main__":
    main()
