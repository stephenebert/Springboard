import os
import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

# existing config
CAPTION_ARRAY   = "coco_caption_texts.npy"
EMBEDDING_ARRAY = "coco_caption_clip.npy"
INDEX_PATH      = "bench_indices/FlatL2.index"     
N_TRIALS        = 1000
K               = 10


# Load data
print(f"Loading captions from '{CAPTION_ARRAY}' and embeddings from '{EMBEDDING_ARRAY}' ...")
texts      = np.load(CAPTION_ARRAY, allow_pickle=True)
embeddings = np.load(EMBEDDING_ARRAY)
print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

# Load index
assert os.path.isfile(INDEX_PATH), f"Index not found: {INDEX_PATH}"
print(f"Loading FAISS index from '{INDEX_PATH}' ...")
index = faiss.read_index(INDEX_PATH)
faiss.omp_set_num_threads(16)

# 1) Measure query latencies
np.random.seed(42)
query_idx   = np.random.choice(len(embeddings), size=N_TRIALS, replace=False)
query_vecs  = embeddings[query_idx]
latencies   = []
for v in query_vecs:
    t0 = time.perf_counter()
    _  = index.search(v.reshape(1, -1), K)
    latencies.append((time.perf_counter() - t0) * 1000)
latencies = np.array(latencies)

# 2) Build and save latency histogram
plt.figure(figsize=(6,4))
plt.hist(latencies, bins=50, edgecolor='black')
plt.title("Query Latency Distribution")
plt.xlabel("Latency (ms)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("latency_hist.png")
plt.close()
print("Saved latency distribution to latency_hist.png")

# 3) Percentile breakdown
perc = np.percentile(latencies, [50, 90, 99, 99.9])
print("\nLatency percentiles (ms):")
for p,v in zip([50,90,99,99.9], perc):
    print(f"  p{p:>5.1f} → {v:6.3f}")

# 4) Distance distribution (within COCO set)
#    pick a small random subset to approximate
sub_idx   = np.random.choice(len(embeddings), size=5000, replace=False)
sub_embs  = embeddings[sub_idx]
# compute pairwise distances to first vector
dists     = np.linalg.norm(sub_embs - sub_embs[0:1], axis=1)

# 5) Build and save distance histogram
plt.figure(figsize=(6,4))
plt.hist(dists, bins=50, edgecolor='black')
plt.title("L2 Distance to First Caption Embed")
plt.xlabel("L2 Distance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("distance_hist.png")
plt.close()
print("Saved distance distribution to distance_hist.png")

# 6) print a small summary table
print("\n## Summary")
results = [
    {"metric": "QPS",      "value": f"{(1/latencies.mean()*1000):.1f}"},
    {"metric": "p50 (ms)", "value": f"{perc[0]:.3f}"},
    {"metric": "p90 (ms)", "value": f"{perc[1]:.3f}"},
    {"metric": "p99 (ms)", "value": f"{perc[2]:.3f}"},
    {"metric": "p99.9(ms)","value": f"{perc[3]:.3f}"}
]
print(tabulate(results, headers="keys", tablefmt="github"))
