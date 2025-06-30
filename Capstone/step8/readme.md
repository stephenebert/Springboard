# Step 8: Scale Your Prototype

This step scales our cross-modal retrieval pipeline to the full training set (≈ 850 K samples) and benchmarks a variety of FAISS indices for large-scale, low-latency deployment. Step 8 brings our prototype from "research mode" to "production scale." In this phase, we embed the entire 850 K-sample corpus into high-dimensional feature vectors (images and captions) using a chunked, out-of-core pipeline that writes directly into HDF5. We then build and benchmark a family of FAISS indices—exact (FlatIP) and approximate (IVF-Flat, IVF-PQ, HNSWFlat) to measure Recall@10, query latency, and RAM footprint at real-world scale. Finally, we sweep critical hyperparameters (nlist, nprobe, M, efSearch) to identify the optimal speed/accuracy trade-offs for deployment. By the end of Step 8, you’ll have a fully scalable retrieval backend tuned for sub-millisecond to low-millisecond performance with minimal memory overhead.

## Repository Structure
```
step8/
├── data/
│ └── metadata.parquet # Unified metadata from Step 7 (local)
├── experiments/ # (local; too large for git)
│ └── full/
│ ├── embeddings_full.h5 # HDF5: all image/text embeddings
│ ├── img_embs_full.npy
│ └── txt_embs_full.npy
├── scale_pipeline_hdf5.py # Chunk & embed → HDF5 (full dataset)
├── evaluate_retrieval.py # Compute exact Recall@K via brute-force
├── faiss_index_comparison.py # Compare FlatIP, IVF-Flat, HNSWFlat, IVF-PQ
├── faiss_param_sweep.py # Hyper-parameter sweep (nlist, nprobe, M, ef)
├── faiss_memory_latency_benchmark.py # RSS vs. latency vs. Recall plotter
├── graphs.py # Regenerate all figures
└── README.md 
```
---

## 1. Chunked Embedding --> HDF5

```bash
python scale_pipeline_hdf5.py \
  --parquet data/metadata.parquet \
  --out experiments/full/embeddings_full.h5 \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --device cpu \
  --batch 64 \
  --chunk 2000
```
Produces embeddings_full.h5 with two datasets:
- /image_embeddings (850668 x 512)
- /text_embeddings (850668 x 512)
  ![Figure 0: Embedding](Figure_0.png)
  - The embeddings took 28 hours to finish running.
## 2. Exact & Approximate Index Benchmark
```bash
python index_benchmark.py \
  --h5 experiments/full/embeddings_full.h5 \
  --nq 1000 \
  --k 10
```
![Figure 1: IVF-Flat (nlist=1024) | Sweep nprobe](Figure_1.png)
- Recall@10 saturates at 82.7 % for nprobe ≥ 2.
- Latency increases from 0.99 ms (nprobe=1) -> 3.56 ms (nprobe=64).
- Takeaway: nprobe = 2-4 offers near-optimal recall with 2-3x speedup over exact.

## 3. Compare Multiple ANN Indices
```bash
python faiss_index_comparison.py \
  --h5 experiments/full/embeddings_full.h5 \
  --nq 1000 \
  --k 10
```
![Figure 2: Approximate Index Comparison](Figure_2.png)
| Index                            | Recall@10 | Latency (ms) |
|----------------------------------|-----------|--------------|
| FlatIP (exact inner-product)     | 82.7 %    | 5.28         |
| IVF-Flat (nlist=1024,nprobe=16)  | 82.7 %    | 2.30         |
| IVF-PQ (nlist=1024,m=64)         | 82.7 %    | 1.03         |
| HNSWFlat (M=32)                  | 80.6 %    | 0.02         |

**Note:** IVF-PQ matches exact recall at ~5x lower latency.

## 4. Memory vs. Latency Trade-off
```bash
python faiss_memory_latency_benchmark.py
```
![Figure 3: RAM Footprint vs. Latency (point size/color proportional to Recall@10)](Figure_3.png)
- RAM footprint ≈ 152 MB for all indices.
- Latency spans 0.02 ms --> 5.28 ms per query.
- Bubble size & color map to Recall@10, highlighting trade-offs:
  -- IVF-PQ & IVF-Flat: best accuracy & moderate speed.
  -- HNSWFlat: ultra-fast but lower recall.
  
## 5. Hyperparameter Sweeps
```bash
python faiss_param_sweep.py \
  --h5 experiments/full/embeddings_full.h5 \
  --nq 1000 \
  --k 10
```
-- Choice for production: IVF-Flat(nlist=1024,nprobe=16) or IVF-PQ.
-- Sweeps over nlist, nprobe, M, efSearch.

  
## 6. Index Benchmarking
Build various FAISS indices and measure their performance based on:
- Recall@10 (exact vs. ANN)
- Avg. latency (ms/query)
- RAM footprint (RSS in MB)

python index_benchmark.py \
  --h5 experiments/full/embeddings_full.h5 \
  --k 10 \
  --nlist 1024 \
  --nprobe_list 1 2 4 8 16 32 64 \
  --hnsw_m_list 8 16 32 \
  --hnsw_ef_list 8 16 32 64 128
  
  **Sample output:**
  
| Index Type            | R@10    | Latency (ms) | RSS (MB) |
|-----------------------|---------|--------------|----------|
| FlatIP                | 82.70%  | 4.56         | 1702     |
| IVF-Flat(nlist=1024,nprobe=16) | 82.70%  | 1.17         | 5087     |
| HNSWFlat(M=32, ef=32) | 81.10%  | 0.03         | 2802     |
| ...                   | ...     | ...          | ...      |
---



# Key Findings
- Exact (FlatIP): Highest recall (82.7 %) but slowest (≈ 5 ms/query).

- IVF-Flat (nlist=1024, nprobe=16): Same recall, 2–3× faster, minimal RAM.

- IVF-PQ: Matches recall & latency (~1 ms) at very low memory.

- HNSWFlat (M=32): Sub-ms queries but recall drops to ~80 %.

- Recommendation: Use IVF-Flat(nlist=1024, nprobe=16) in production for optimal speed-accuracy trade-off.

