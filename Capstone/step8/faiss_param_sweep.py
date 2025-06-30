"""
faiss_param_sweep.py

Evaluate various FAISS index hyperparameters on your full CLIP embeddings:
  - measures Recall@K, average latency, and RSS memory
  - sweeps IVF-Flat (nlist / nprobe) and HNSWFlat (M / efSearch)
"""

import time
import psutil
import h5py
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("FAISS not installed : falling back to NumPy brute-force for ground truth")

# Path to your full-image-embeddings HDF5
H5_PATH = r"C:\Users\steph\OneDrive\Desktop\step7\experiments\full\embeddings_full.h5"

# Number of random queries and recall@K
NQ = 1000
K  = 10

# IVF-Flat settings to sweep
IVF_LISTS   = [512, 1024, 2048]
IVF_PROBES  = [8, 16, 32]

# HNSWFlat settings to sweep
HNSW_Ms       = [16, 32]
HNSW_efSearch = [16, 32, 64]

# LOAD EMBEDDINGS

print(f"Loading embeddings from {H5_PATH}")
with h5py.File(H5_PATH, "r") as hf:
    xb = hf["image_embeddings"][:]  # shape (N, D)
print(f" loaded {xb.shape[0]:,} vectors; dim = {xb.shape[1]}")

# Normalize (inner-product similarity)
xb = xb.astype("float32")
faiss.normalize_L2(xb)

# sample some queries
np.random.seed(0)
qs = xb[np.random.choice(len(xb), size=NQ, replace=False)]

# compute exact ground-truth indices for recall (only if FAISS available)
if HAS_FAISS:
    # build a FlatIP index for ground truth
    print("\nBuilding FlatIP for exact ground truth...", end="", flush=True)
    flat_gt = faiss.IndexFlatIP(xb.shape[1])
    flat_gt.add(xb)
    idx_gt = flat_gt.search(qs, K)[1]
    print(" done")
else:
    idx_gt = None  

proc = psutil.Process()
results = []

def measure(index, name):
    # measure RSS before search
    mem0 = proc.memory_info().rss / 1024**2
    # time a warm-up + real search
    _ = index.search(qs[:50], K) 
    t0 = time.time()
    D, I = index.search(qs, K)
    latency = (time.time() - t0) / NQ * 1000  # ms/query
    # measure RSS after
    mem1 = proc.memory_info().rss / 1024**2
    # compute recall@K
    if idx_gt is not None:
        recall = 100 * np.mean((I == idx_gt).any(axis=1))
    else:
        recall = float("nan")
    print(f"{name:30s}  R@{K} = {recall:5.2f}%  | {latency:6.2f} ms  | {mem1-mem0:5.1f} MB")
    results.append({
        "index":      name,
        f"R@{K}":     recall,
        "latency_ms": latency,
        "memory_mb":  mem1 - mem0,
    })

#SWEEP IVF-Flat

if HAS_FAISS:
    d = xb.shape[1]
    for nlist in IVF_LISTS:
        quantizer = faiss.IndexFlatIP(d)
        ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
        ivf.train(xb)
        ivf.add(xb)
        for nprobe in IVF_PROBES:
            ivf.nprobe = nprobe
            measure(ivf, f"IVF-Flat(nlist={nlist}, nprobe={nprobe})")

    # SWEEP HNSWFlat
    for M in HNSW_Ms:
        hnsw = faiss.IndexHNSWFlat(d, M)
        hnsw.hnsw.efConstruction = 100
        hnsw.add(xb)
        for ef in HNSW_efSearch:
            hnsw.hnsw.efSearch = ef
            measure(hnsw, f"HNSWFlat(M={M}, efSearch={ef})")


else:
    # fallback: brute-force cosine by NumPy
    print("\nNumPy brute-force recall & latency:")
    t0 = time.time()
    sims = qs @ xb.T
    I = np.argpartition(-sims, K-1, axis=1)[:, :K]
    latency = (time.time() - t0) / NQ * 1000
    recall = 100 * np.mean((I == idx_gt).any(axis=1)) if idx_gt is not None else np.nan
    mem1 = proc.memory_info().rss / 1024**2
    mem0 = mem1  # no change tracked here
    results.append({
        "index":      "Brute-force",
        f"R@{K}":     recall,
        "latency_ms": latency,
        "memory_mb":  0.0,
    })
    print(f"Brute-force : R@{K} = {recall:5.2f}% | {latency:6.2f} ms |   — MB")

#REPORT SUMMARY

import pandas as pd
df = pd.DataFrame(results)
print("\nSummary:")
print(df.to_string(index=False))
