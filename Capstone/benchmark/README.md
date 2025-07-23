# COCO Caption ANN Benchmark

Efficient similarity search at scale is the backbone of many modern AI systems. From AI‑powered image search and recommendation engines to real‑time language retrieval and conversational agents. Yet choosing the right ANN index often involves a trade‑off between accuracy, build time, memory footprint, and query throughput. By quantitatively benchmarking FlatL2 against IVF variants on a realistic COCO‑caption embedding workload, we can identify which index delivers near‑perfect recall at orders‑of‑magnitude faster query speeds and modest build overhead, empowering practitioners to architect production‑grade pipelines that serve millions of queries per second without sacrificing result quality.

---

## Index Primer

1. **FlatL2**  
   - "Flat" means no clustering or partitioning. Every vector lives in one big array.  
   - L2 (Euclidean) distance to every vector → exact recall, O(N) per query.  
   - **Pros**: trivial to build, 100% recall.  
   - **Cons**: very slow at scale (limited QPS), large memory bandwidth.

2. **IVF (Inverted File / `IndexIVFFlat`)**  
   - Cluster N vectors into `nlist` cells via k‑means → assign each vector to its nearest centroid.  
   - At query time, search only the top `nprobe` cells, then do exact L2 within those lists.  
   - **Pros**: orders‑of‑magnitude higher QPS for tiny recall loss.  
   - **Cons**: build time for clustering, storage for centroids & lists, need to tune `nlist` & `nprobe`.

---

## Contents
``` bash
benchmark/
├── coco_caption_texts.npy      # 591 753 raw captions
├── coco_caption_clip.npy       # 591 753 × 512 CLIP embeddings
├── generate_coco_texts.py      # extract & save .npy captions
├── generate_coco_embeds.py     # encode & save .npy embeddings
├── benchmark_ann.py            # build/query FAISS indices & print summary
├── extended_metrics.py         # compute percentiles & histograms
├── latency_hist.png            # query latency distribution
├── distance_hist.png           # L2 distance distribution
└── bench_indices/              # (output) FlatL2.index, IVF_1024.index, IVF_4096.index
```

---

## 1. Prepare the data

1. **Extract raw captions**
```bash
   cd benchmark
   python generate_coco_texts.py \
     --ann_path /path/to/annotations/captions_train2017.json \
     --out_texts coco_caption_texts.npy
  ```
2. **Encode with CLIP**
```bash
python generate_coco_embeds.py \
  --texts coco_caption_texts.npy \
  --out_embeds coco_caption_clip.npy \
  --device mps   # or cuda/cpu
```
## 2. Run the benchmarks
``` bash
python benchmark_ann.py \
  --texts    coco_caption_texts.npy \
  --embeds   coco_caption_clip.npy \
  --out_dir  bench_indices
```
This does:
1. Load 591,753 caption strings and their 512‑dim CLIP embeddings.
2. Build three FAISS indices:
  - FlatL2
  - IVF₁₀₂₄, nlist=1024
  - IVF₄₀₉₆, nlist=4096
3. For each index:
   - Train (if needed) and add all vectors
   - Measure build time and on‑disk size
   - Query 1000 random captions and report: Mean latency (ms) --> QPS and Recall@1,5,10 against exact FlatL2
     
Finally, it prints out a table:

| index     | build\_s | size\_MB | lat\_ms |     QPS |  R\@1 |  R\@5 | R\@10 |
| :-------- | -------: | -------: | ------: | ------: | :---: | :---: | :---: |
| FlatL2    |     0.07 |  1155.77 |    0.31 |   3 205 | 0.948 | 0.981 | 0.990 |
| IVF\_1024 |     0.49 |  1162.29 |    0.01 | 160 486 | 0.948 | 0.981 | 0.990 |
| IVF\_4096 |     1.71 |  1168.31 |    0.00 | 290 183 | 0.948 | 0.981 | 0.990 |

## 3. Tail‑Latency Analysis

![Latency Distribution](latency_hist.png)

- FAISS will use up to 16 threads.
- Loading captions from 'coco_caption_texts.npy' and embeddings from 'coco_caption_clip.npy' …
- Saved latency distribution to latency_hist.png

Latency percentiles (ms):
  p50   → 17.012
  p90   → 17.476
  p99   → 19.517
  p99.9 → 20.778

- Median (p50): 17.0 ms

- 90th pct.: 17.5 ms

- 99th pct.: 19.5 ms

- 99.9th pct.: 20.8 ms
  
## 4. Embedding Distance Distribution

![Distance Distribution](distance_hist.png)

- Loaded 591,753 embeddings of dimension 512
- Saved distance distribution to distance_hist.png

Most caption embeddings lie between 22–30 L2 distance from an arbitrary reference—indicating a fairly tight shell. Very few lie outside [15, 32], informing appropriate `nprobe` and quantization granularity.


## 5. Key Findings
- FlatL2: exact recall but limited ~3 k QPS → useful for small datasets or offline analysis.

- IVF_1024: sweet spot → 50× speed‑up with zero recall loss.

- IVF_4096: peak throughput (~290 k QPS) → ideal for high‑concurrency production.

- Tail metrics: ensure SLA‑compliance by tuning for your p99/p99.9 budgets.

- Distance histograms: guide hyperparameter choices (cluster count, subquantization).

Recommendation and observation:

- For max throughput, use IVF_4096.

- For fast build + good speed, IVF_1024 is a sweet spot.

- Use FlatL2 only for prototyping or small‐scale demos.

## 6. Requirements
``` bash
faiss-cpu         # or faiss-gpu
numpy
tabulate
tqdm
```
Install via:
``` bash
pip install faiss-cpu numpy tabulate tqdm
```
## 7. Reproducibility
- All random seeds are fixed for the query sample.

- Benchmarks run on 16 threads (```OMP_NUM_THREADS=16```).

- Scripts accept ```--device``` flags to leverage MPS/CUDA.
