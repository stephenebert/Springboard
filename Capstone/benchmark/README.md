# COCO Caption ANN Benchmark

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
├── coco_caption_texts.npy # Extracted COCO captions (591 753 strings)
├── coco_caption_clip.npy # CLIP embeddings for each caption
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





