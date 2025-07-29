
# retrieval-backend

This folder contains everything needed to build, index, and benchmark text‐based retrieval pipelines over COCO captions, including:

- **Embeddings**: CLIP‐ViT or SBERT encoders  
- **FAISS indices**: HNSW, IVF‑Flat, IVF‑PQ  
- **Evaluation**: baseline (FAISS only) and GPT‑4o‐vision log‑prob reranker  

---
## Directory Structure
```
bash
retrieval-backend/
├── data/
│ ├── raw/
│ │ └── captions_val2017.json # COCO captions
│ └── embeds_val2017.npy # precomputed embeddings
├── figs/
│ ├── cost_metrics.png # recall vs cost & scale
│ └── pipeline_comparison.png # final pipeline bar chart
├── models/
│ └── … # downloaded MM‑Embed/SBERT model files
├── embed_coco.py # caption→embedding
├── encoder_clip.py # CLIP embedding wrapper
├── encoder_sbert.py # SBERT embedding wrapper
├── index_builder.py # build FAISS index
├── evaluate_baseline.py # FAISS-only benchmark
├── evaluate_reranked.py # two-stage + GPT‑4o reranker
├── requirements.txt
└── README.md
```



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

## Results
### Cost vs Scale (Reranker experiments)
| limit | top\_k | Recall\@1 | Latency (ms/q) | Cost US\$ (est) |
| :---: | :----: | :-------: | :------------: | :-------------: |
|  300  |   10   |   0.9033  |      616.9     |       0.35      |
|  1000 |   10   |   0.8810  |      565.3     |       1.44      |
|  3000 |   15   |   0.8567  |      586.8     |       5.10      |
|  600  |   10   |   0.8767  |      591.1     |       0.86      |
|  1500 |   10   |   0.8853  |      571.9     |       2.16      |
|  2000 |   10   |   0.8650  |      585.0     |       2.88      |
|  1000 |    5   |   0.8970  |      563.3     |       0.72      |
|  1000 |   15   |   0.8660  |      582.2     |       1.44      |
|  4000 |   10   |   0.8692  |      585.8     |       5.76      |


### Pipeline Comparison @ 5 000 captions, top_k = 1


|               Pipeline              | Recall\@1 | Latency (ms/q) | Cost (US\$) |
| :---------------------------------: | :-------: | :------------: | :---------: |
|         CLIP + HNSW (ef=64)         |   1.0000  |      65.2      |     0.72    |
|       MM‑Embed + HNSW (ef=64)       |   0.9974  |      75.7      |     0.72    |
| MM‑Embed + IVF‑PQ (nlist=512, m=32) |   0.9958  |      70.2      |     0.72    |





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

