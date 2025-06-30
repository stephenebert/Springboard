# Step 8: Scale Your Prototype

This step scales our cross-modal retrieval pipeline to the full training set (≈ 850 K samples) and benchmarks approximate-nearest-neighbor (ANN) indices for large-scale deployment.

---

## Repository Structure
```
step8/
├── data/                  # Symlink or local data directory
│   └── metadata.parquet   # Unified metadata from Step 7
├── experiments/
│   └── full/              # Outputs for full-scale embeddings & indices
│       ├── embeddings_full.h5 # HDF5 file with all image/text embeddings
│       └── index_benchmarks/  # FAISS index artifacts & logs
├── scale_pipeline_hdf5.py # Chunked embedding --> HDF5 (full dataset)
├── index_benchmark.py     # Build FAISS indices, measure recall/latency/RSS
├── faiss_param_sweep.py   # Hyperparameter sweep for IVF-Flat & HNSWFlat
├── requirements.txt       # Step 8 specific deps (h5py, faiss-cpu, psutil)
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

Chunked I/O via HDF5 (with LZF compression) keeps RAM usage constant.
Transforms match those in Step 7 for consistency.
