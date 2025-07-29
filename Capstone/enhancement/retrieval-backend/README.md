
# retrieval-backend

This folder contains everything needed to build, index, and benchmark text‐based retrieval pipelines over COCO captions, including:

- **Embeddings**: CLIP‐ViT or SBERT encoders  
- **FAISS indices**: HNSW, IVF‑Flat, IVF‑PQ  
- **Evaluation**: baseline (FAISS only) and GPT‑4o‐vision log‑prob reranker  

---

## Table of Contents

- [Installation](#installation)  
- [Data](#data)  
- [Scripts](#scripts)  
  - [`embed_coco.py`](#embed_cocopy)  
  - [`encoder_clip.py` / `encoder_sbert.py`](#encoders)  
  - [`index_builder.py`](#index_builderpy)  
  - [`evaluate_baseline.py`](#evaluate_baselinepy)  
  - [`evaluate_reranked.py`](#evaluate_rerankedpy)  
- [Usage Examples](#usage-examples)  
- [Environment Variables](#environment-variables)  
- [License](#license)  

---

## Installation

```bash
git clone https://github.com/<YOUR_ORG>/capstone-enhancement.git
cd capstone-enhancement/retrieval-backend

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Data
Place your COCO validation captions JSON here:

``` bash
data/raw/captions_val2017.json
```

## Scripts

### ```embed_coco.py```

Load up to N COCO captions, embed them in batches, and save to a ```.npy```
``` bash
python embed_coco.py \
  --json /path/to/captions_val2017.json \
  --out data/embeds_val2017.npy \
  --limit 5000 \
  --batch 128 \
  --encoder clip   \
  --verbose
```
- ```--encoder {clip,sbert}```: which text embedder to use.
- ```--limit / --batch```: subset size & batch size.
### ```encoder_clip.py / encoder_sbert.py```
Thin wrappers exposing a single function:
``` bash
from encoder_clip import embed_texts
vs.
from encoder_sbert import embed_texts
```
Use whichever fits your accuracy / speed trade‑off.

### ```index_builder.py```

Build a FAISS index from your precomputed ```.npy``` embeddings:

``` bash
# HNSW
python index_builder.py \
  --embeds data/embeds_val2017.npy \
  --out hnsw_val2017.fai \
  --engine hnsw \
  --ef_search 64

# IVF‑PQ
python index_builder.py \
  --embeds data/embeds_val2017.npy \
  --out ivfpq_val2017.fai \
  --engine ivfpq \
  --nlist 512 \
  --pq_m 32
```
- ```--engine {hnsw, ivf_flat, ivfpq}```
- HNSW: tune ```--ef_search```
- IVF‑PQ: set ```--nlist, --pq_m```
### ```evaluate_baseline.py```
Single-stage retrieval evaluation (Recall@K + latency):
``` bash
python evaluate_baseline.py \
  --json  /path/to/captions_val2017.json \
  --engine hnsw \
  --index  hnsw_val2017.fai \
  --limit 5000 \
  --top_k 1 \
  --batch 256 \
  --ef_search 64
```
Outputs:
- Recall@K
- average ms/query

### ```evaluate_reranked.py```
``` bash
export OPENAI_API_KEY=sk-...
python evaluate_reranked.py \
  --json  /path/to/captions_val2017.json \
  --limit 1000 \
  --top_k 10
```
Outputs:
- Reranked Recall@1
- avg latency per query

## Usage Examples
All of the above get wired into Cost metrics.ipynb, which:

1. Reads your log files

2. Builds a Pandas table of (limit, top_k, recall, latency, cost)

3. Plots recall vs cost curves

## Environment Variables

``` bash
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY=sk-...
```

