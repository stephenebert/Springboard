import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

try:
    import faiss
except ImportError:
    raise ImportError("faiss not found. pip install faiss-cpu")

# load embeddings
H5_PATH = "experiments/full/embeddings_full.h5"
with h5py.File(H5_PATH, "r") as hf:
    xb = hf["image_embeddings"][:]     # (N, D)
print(f"Loaded xb shape = {xb.shape}")

# sample query set
np.random.seed(0)
nq = 1000
idx_q = np.random.choice(len(xb), nq, replace=False)
xq = xb[idx_q].astype("float32")
gt = idx_q 

# helper: evaluate recall & latency
def benchmark_index(index, x_add, x_q, k=10):
    """Train/add (if needed), then search x_q; return (recall10, latency_ms)."""
    if not index.is_trained:
        index.train(x_add)
    index.add(x_add)
    # warm-up
    _D, _I = index.search(x_q[:5], k)
    t0 = time.perf_counter()
    D, I = index.search(x_q, k)
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000.0 / len(x_q)
    hits = np.array([gt[i] in I[i] for i in range(len(x_q))])
    recall10 = hits.mean() * 100.0
    return recall10, latency

# Flat (exact) index
d = xb.shape[1]
flat = faiss.IndexFlatIP(d)
r_flat, t_flat = benchmark_index(flat, xb.astype("float32"), xq)
print(f"FlatIP : R@10 = {r_flat:.2f}% : {t_flat:.3f} ms/q")

# IVF-Flat
nlist = 1024
ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)

ivf.nprobe = 16
r_ivf, t_ivf = benchmark_index(ivf, xb.astype("float32"), xq)
print(f"IVF-Flat (nlist={nlist}, nprobe={ivf.nprobe})  : R@10 = {r_ivf:.2f}% : {t_ivf:.3f} ms/q")

# HNSWFlat
# M=32 is a common choice; higher M -> better recall but slower search.
M = 32
hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
r_hnsw, t_hnsw = benchmark_index(hnsw, xb.astype("float32"), xq)
print(f"HNSWFlat (M={M})  : R@10 = {r_hnsw:.2f}% : {t_hnsw:.3f} ms/q")

# IVF-PQ
nlist_pq = 1024
m_pq = 64     
nbits = 8     
ivfpq = faiss.IndexIVFPQ(
    faiss.IndexFlatIP(d), d,
    nlist_pq, m_pq, nbits,
    faiss.METRIC_INNER_PRODUCT
)
ivfpq.nprobe = 16
r_pq, t_pq = benchmark_index(ivfpq, xb.astype("float32"), xq)
print(f"IVF-PQ (nlist={nlist_pq}, m={m_pq}, nprobe={ivfpq.nprobe})  : R@10 = {r_pq:.2f}% : {t_pq:.3f} ms/q")

labels = [
    f"FlatIP\n({r_flat:.1f}%, {t_flat:.2f}ms)",
    f"IVF-Flat\n({r_ivf:.1f}%, {t_ivf:.2f}ms)",
    f"HNSWFlat\n({r_hnsw:.1f}%, {t_hnsw:.2f}ms)",
    f"IVF-PQ\n({r_pq:.1f}%, {t_pq:.2f}ms)",
]
recalls = [r_flat, r_ivf, r_hnsw, r_pq]
times   = [t_flat, t_ivf, t_hnsw, t_pq]

plt.figure(figsize=(6,4))
plt.scatter(times, recalls)
for x,y,label in zip(times, recalls, labels):
    plt.text(x, y, label, ha="left", va="bottom", fontsize=8)
plt.xlabel("Latency per query (ms)")
plt.ylabel("Recall@10 (%)")
plt.title("Approximate Index Comparison")
plt.grid(True)
plt.tight_layout()
plt.show()
