# COCO Caption ANN Benchmark

Efficient similarity search at scale is the backbone of many modern AI systems—from AI‐powered image search and recommendation engines to real‐time language retrieval and conversational agents yet choosing the right ANN index often involves a trade‑off between accuracy, build time, memory footprint, and query throughput. By quantitatively benchmarking FlatL2 against IVF variants on a realistic COCO‐caption embedding workload, we can identify which index delivers near‑perfect recall at orders‑of‑magnitude faster query speeds and modest build overhead, empowering practitioners to architect production‑grade pipelines that serve millions of queries per second without sacrificing result quality.

This repo contains everything you need to reproduce a quantitative comparison of FAISS approximate‑nearest‑neighbor (ANN) indices on COCO caption embeddings. We benchmark:

- **IndexFlatL2** (exact L2 search)  
- **IndexIVFFlat** with IVF₁₀₂₄ and IVF₄₀₉₆ centroids  

Measuring for each index:
- **Build time** (seconds)  
- **On‑disk size** (MB)  
- **Query latency** (ms/query) & **QPS** (queries/sec)  
- **Recall@k** (k = 1, 5, 10)  

All runs were performed on an M4 series chip (16 CPU threads).

---

## Contents
``` bash
benchmark/
├── coco_caption_texts.npy # Extracted COCO captions (591 753 strings; stored locally due to size)
├── coco_caption_clip.npy # CLIP embeddings for each caption (stored locally due to size)
├── generate_coco_texts.py # Script to extract & save coco_caption_texts.npy
├── generate_coco_embeds.py # Script to encode & save coco_caption_clip.npy
├── benchmark_ann.py # Runs FAISS benchmarks & prints summary table
└── bench_indices/ # (empty; benchmarks save indexes here)
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
   - Train (if needed) & add all vectors
   - Measure build time & on‑disk size
   - Query 1000 random captions and report: Mean latency (ms) --> QPS and Recall@1,5,10 against exact FlatL2
     
Finally, it prints out a table:

| index     | build\_s | size\_MB | lat\_ms |     QPS |  R\@1 |  R\@5 | R\@10 |
| :-------- | -------: | -------: | ------: | ------: | :---: | :---: | :---: |
| FlatL2    |     0.07 |  1155.77 |    0.31 |   3 205 | 0.948 | 0.981 | 0.990 |
| IVF\_1024 |     0.49 |  1162.29 |    0.01 | 160 486 | 0.948 | 0.981 | 0.990 |
| IVF\_4096 |     1.71 |  1168.31 |    0.00 | 290 183 | 0.948 | 0.981 | 0.990 |

## 3. Key Findings
- Exact FlatL2 is trivial to build (0.07 s) but only ~3 k QPS.
- IVF_1024 yields a 50 times speed‑up (~160 k QPS) with no recall loss.
- IVF_4096 pushes throughput to ~290 k QPS—ideal for high‑QPS production.
- Index sizes remain ~1.1 GB, and recall@k is identical across all settings.

Recommendation and observation:

- For max throughput, use IVF_4096.

- For fast build + good speed, IVF_1024 is a sweet spot.

- Use FlatL2 only for prototyping or small‐scale demos.

## 4. Requirements
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
## 5. Reproducibility
- All random seeds are fixed for the query sample.

- Benchmarks run on 16 threads (```OMP_NUM_THREADS=16```).

- Scripts accept ```--device``` flags to leverage MPS/CUDA.




