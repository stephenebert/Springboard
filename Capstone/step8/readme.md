# Step 8: Scale Your Prototype

This step scales our cross-modal retrieval pipeline to the full training set (≈ 850 K samples) and benchmarks approximate-nearest-neighbor (ANN) indices for large-scale deployment.

---

## Repository Structure
```
step8/
├── data/
│ └── metadata.parquet # Unified metadata from Step 7 and locally stored on computer
├── experiments/
│ ├── full/
│ │ ├── embeddings_full.h5 # HDF5: all image/text embeddings and locally stored on computer
│ │ ├── img_embs_full.npy # locally stored on computer
│ │ └── txt_embs_full.npy # locally stored on computer
│ └── index_benchmarks/ # FAISS index artifacts & logs
├── scale_pipeline_hdf5.py # Chunked embed → HDF5 (full dataset)
├── index_benchmark.py # Build FAISS indices, measure recall/latency/RSS
├── faiss_param_sweep.py # Hyperparameter sweep for IVF-Flat & HNSWFlat
├── faiss_index_comparison.py # Compare FlatIP, IVF-Flat, HNSWFlat, IVF-PQ
├── faiss_memory_latency_benchmark.py # RSS vs. latency vs. recall plotter
├── evaluate_retrieval.py # Ground-truth recall via brute-force
├── graphs.py # Regenerate all figures
└── README.md
```
---

## Setup & Installation

1. **Clone & enter directory**

   ```bash
   git clone <repo-url> step8
   cd step8
   
2. **Install dependencies**

    ```bash
   conda create -n step8 python=3.10
   conda activate step8
   pip install -r requirements.txt

3. Prepare metadata – ensure data/metadata.parquet exists (from Step 7).

# Full-Scale Embedding

Generate 512-D image/text embeddings for all ≈ 850 K training samples and save them into a single HDF5 file:

```bash
python scale_pipeline_hdf5.py \
    --parquet data/metadata.parquet \
    --out experiments/full/embeddings_full.h5 \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --batch 64 \
    --chunk 2000 \
    --workers 4
```

- Chunked I/O via HDF5 (with LZF compression) keeps RAM usage constant.
- Transforms match those in Step 7 for consistency.
  
# Index Benchmarking
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
# Hyperparameter Sweep

Run a hyperparameter sweep to find the optimal balance between recall, latency, and memory for IVF-Flat and HNSWFlat indices.

---
```bash
   python faiss_param_sweep.py \
     --h5 experiments/full/embeddings_full.h5 \
     --k 10 \
     --nlist_list 512 1024 2048 \
     --nprobe_list 8 16 32 \
     --hnsw_m_list 16 32 \
     --hnsw_ef_list 16 32 64
```
---

**Generates a summary table:**

| Index                 | R@10    | Latency (ms) | Memory (MB) |
|-----------------------|---------|--------------|-------------|
| IVF-Flat(nlist=1024,16)| 82.10%  | 1.72         | 0.13        |
| HNSWFlat(M=32, ef=64) | 29.70%  | 0.02         | 0.00        |
| ...                   | ...     | ...          | ...         |


# Key Findings
- Exact (FlatIP): Provides the highest recall (82.7%) but has high latency at ~4.6 ms/query.

- IVF-Flat (nlist=1024, nprobe=16): Matches the exact recall of the FlatIP index while being significantly faster at ~1.7 ms/query.

- HNSWFlat (M=32, ef=32–64): Achieves sub-millisecond query times but at the cost of a significant drop in recall to ~30%.

- Memory Usage: The IVF-Flat index demonstrates a favorable trade-off, using about 0.1 MB per 10,000 vectors for its substantial latency improvements.

- Recommendation: Use IVF-Flat(nlist=1024, nprobe=16) for production environments. It delivers the same accuracy as an exact search with a 2-3x speedup and minimal memory overhead.

