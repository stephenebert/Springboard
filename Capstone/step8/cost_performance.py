import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

try:
    import faiss
except ImportError:
    raise ImportError("faiss not found. Install it with `pip install faiss-cpu`")

# Load full‐dataset image embeddings from HDF5
H5_PATH = "experiments/full/embeddings_full.h5"
with h5py.File(H5_PATH, "r") as hf:
    xb = hf["image_embeddings"][:]    # shape (N, D)
print(f"Loaded embeddings matrix xb with shape {xb.shape}")

# Sample a fixed set of 1,000 queries (no replacement)
np.random.seed(0)
nq = 1000
query_idx = np.random.choice(len(xb), size=nq, replace=False)
xq = xb[query_idx].astype("float32")
true_ids = query_idx  # for each query i, the "ground‐truth" match is xb[true_ids[i]]

# Build & train an IVFFlat index (inner‐product)
d = xb.shape[1]
nlist = 1024
quantizer = faiss.IndexFlatIP(d)  # quantizer uses inner-product
# pass metric as a positional argument instead of keyword
ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

assert not ivf.is_trained
ivf.train(xb.astype("float32"))   # must train before adding
assert ivf.is_trained
ivf.add(xb.astype("float32"))     # build the index
print("IVFFlat index (nlist=1024) trained & loaded with data")

# Sweep nprobe and record recall@10 + latency
k = 10
probes = [1, 2, 4, 8, 16, 32, 64]
recalls = []
latencies = []

for p in probes:
    ivf.nprobe = p
    t0 = time.perf_counter()
    D_ivf, I_ivf = ivf.search(xq, k)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0 / nq
    hit = np.array([true_ids[i] in I_ivf[i] for i in range(nq)])
    rec10 = hit.mean() * 100.0

    print(f"nprobe={p:>2} | Recall@{k} = {rec10:5.2f}% | Latency = {latency_ms:6.3f} ms")
    recalls.append(rec10)
    latencies.append(latency_ms)

# Plot Recall@10 vs Latency
plt.figure(figsize=(6,4))
plt.plot(latencies, recalls, marker="o")
for lp, rc, p in zip(latencies, recalls, probes):
    plt.text(lp, rc, f" p={p}", ha="left", va="bottom")

plt.xlabel("Latency per query (ms)")
plt.ylabel(f"Recall@{k} (%)")
plt.title(f"IVF‐Flat (nlist={nlist})  •  Sweep nprobe")
plt.grid(True)
plt.tight_layout()
plt.show()
