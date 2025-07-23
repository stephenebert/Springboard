# COCO Caption ANN Benchmark

This repo contains everything you need to reproduce a quantitative comparison of FAISS approximate‑nearest‑neighbor (ANN) indices on COCO caption embeddings. We benchmark:

- **IndexFlatL2** (exact L2 search)  
- **IndexIVFFlat** with IVF₁₀₂₄ and IVF₄₀₉₆ centroids  

Measuring for each index:
- **Build time** (seconds)  
- **On‑disk size** (MB)  
- **Query latency** (ms/query) & **QPS** (queries/sec)  
- **Recall@k** (k = 1, 5, 10)  

All runs were performed on an Apple M4 Max (16 CPU threads).

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
