#!/usr/bin/env python3
# benchmark_ann.py
# Place in benchmark/ and run: python benchmark_ann.py

import os, time, numpy as np, faiss, multiprocessing
from tabulate import tabulate

# File names in this folder 
CAPTION_ARRAY   = "coco_caption_texts.npy"
EMBEDDING_ARRAY = "coco_caption_clip.npy"
OUTPUT_DIR      = "bench_indices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FAISS THREADS
n_threads = min(16, multiprocessing.cpu_count())
faiss.omp_set_num_threads(n_threads)
print(f"[*] FAISS using {n_threads} threads")

# Load data
print(f"[*] Loading captions from '{CAPTION_ARRAY}' and embeddings from '{EMBEDDING_ARRAY}'")
captions  = np.load(CAPTION_ARRAY, allow_pickle=True)
embeds    = np.load(EMBEDDING_ARRAY)
n, d      = embeds.shape
print(f"[+] {n:,} embeddings of dimension {d}")

# ——— Index configurations ————————————————————————————————————
index_confs = [
    ("FlatL2",   lambda: faiss.IndexFlatL2(d)),
    ("IVF_1024", lambda: faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 1024, faiss.METRIC_L2)),
    ("IVF_4096", lambda: faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 4096, faiss.METRIC_L2)),
]

results = []
for name, make_idx in index_confs:
    print(f"\n=== {name} ===")
    idx = make_idx()

    # train if IVF
    if isinstance(idx, faiss.IndexIVF):
        t0 = time.time()
        idx.train(embeds)
        print(f" trained in {time.time() - t0:.2f}s")

    # add
    t0 = time.time()
    idx.add(embeds)
    build_t = time.time() - t0
    print(f" added {n:,} vectors in {build_t:.2f}s")

    # write to disk
    path   = os.path.join(OUTPUT_DIR, f"{name}.index")
    faiss.write_index(idx, path)
    sizeMB = os.path.getsize(path) / (1024**2)
    print(f" file size: {sizeMB:.2f} MB")

    # query performance & recall
    nq = 1_000
    np.random.seed(0)
    qidx = np.random.choice(n, nq, replace=False)
    queries = embeds[qidx]

    idx.search(queries[:5], 10)  # warmup
    t0 = time.time()
    D, I = idx.search(queries, 10)
    qt = time.time() - t0

    lat_ms = (qt / nq) * 1000
    qps    = nq / qt
    print(f" latency/query: {lat_ms:.2f} ms  (QPS: {qps:.1f})")

    recalls = {}
    for k in (1,5,10):
        hits = sum(1 for i, true in enumerate(qidx) if true in I[i,:k])
        recalls[k] = hits / nq
        print(f" recall@{k}: {recalls[k]:.3f}")

    results.append({
        "index": name,
        "build_s": f"{build_t:.2f}",
        "size_MB": f"{sizeMB:.2f}",
        "lat_ms": f"{lat_ms:.2f}",
        "QPS": f"{qps:.1f}",
        **{f"R@{k}": f"{recalls[k]:.3f}" for k in recalls}
    })

# ——— Print summary ————————————————————————————————————————————
print("\n## Summary\n")
print(tabulate(results, headers="keys", tablefmt="github"))
