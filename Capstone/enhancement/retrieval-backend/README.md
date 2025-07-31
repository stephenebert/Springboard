
# retrieval-backend

This folder contains everything needed to build, index, and benchmark text‐based retrieval pipelines over COCO captions, including:

- **Embeddings**: CLIP-ViT, SBERT, MM-Embed, **ImageBind**  
- **FAISS indices**: HNSW, IVF-Flat, IVF-PQ  
- **Evaluation**: baseline (FAISS only) and GPT‑4o‐vision log‑prob reranker  


| Stage | Options |
|-------|---------|
| **Encoders** | <br>• CLIP-ViT/B32  (384-D)<br>• SBERT-all-MiniLM  (384-D)<br>• MM-Embed-Base  (1 024-D)<br>• **ImageBind-Huge**  (1 024-D, 8-branch multimodal) |
| **FAISS engines** | HNSW  (exact, RAM-friendly)<br>IVF-Flat  (shards, RAM-heavy, fast)<br>IVF-PQ  (shards + product-quantisation, smallest) |
| **Evaluation** | • Baseline Recall@K + latency<br>• GPT-4o-vision reranker † |
| **Plots** | recall vs cost, recall vs latency, pipeline bars |

† *Reranker uses real GPT-4o API calls – costs shown are true token spend.*


## Highlights

- **Data to Embeddings**: You take the COCO validation captions JSON and turn each caption into a high‑dimensional vector, using either OpenAI’s CLIP‐Vision text encoder or NVIDIA’s multimodal MM‑Embed (via Hugging Face).
- **Embeddings to FAISS index**: Those vectors get shoved into FAISS, using either:
 1. HNSW, a graph‑based structure that gives exact (or near‑exact) neighbors very quickly, or
 2. IVF‑PQ, a two‑stage (inverted‐file + product quantization) scheme that trades a tiny bit of accuracy for memory and speed gains.
- **Index to Queries to Metrics**:
  You then query each caption against the index (i.e. "can I retrieve the exact same caption as its own nearest neighbor?"), measure:
  1. Recall@1 (did the top result match the query?)
  2. Latency per query
  3. Estimated cost (rough token‑count x price estimate for the reranker experiments)
- **Evaluation**  
  1. Reranker: GPT‑4o‑vision (model: `gpt-4o-instruct`). Costed at \$0.000144 per 128‑token prompt (we consumed real API credits)  
  2. Recall@K & avg ms/query for 5000 COCO captions  
   3. Cost bars in our plots reflect the **actual OpenAI credits** spent




## Takeaways
1. For pure speed+accuracy: stick with CLIP vectors + HNSW (easy, memory‑light, perfect recall).
2. If you want higher‑dim multi‑modal features (ImageBind/MM‑Embed) you still get excellent recall (> 0.995) at ≈ 70 ms/query—either with HNSW or the more memory‑efficient IVF‑PQ.
3. Reranker experiments reveal how cost and latency scale as you tune how many candidates you rerank (limit, top_k).

---
## Directory Structure
```bash
retrieval-backend/
├── models/ # local-only — FAISS indexes & pickles
│ ├── hnsw_val2017.faiss # CLIP+HNSW (5 k vectors, 384‑D)
│ ├── ivfpq_val2017.faiss # CLIP+IVF‑PQ (5 k vectors)
│ ├── hnsw_mmembed.faiss # MM‑Embed+HNSW (5 k, 1 024‑D)
│ └── ivfpq_mmembed.faiss # MM‑Embed+IVF‑PQ
├── data/ # local-only — COCO captions & embeddings
│ ├── captions_val2017.json # COCO val split
│ ├── embeds_val2017.npy # CLIP‐based embeddings
│ └── embeds_mmembed.npy # MM‑Embed embeddings
├── embed_coco.py # JSON → batched .npy embeddings
├── index_builder.py # .npy → FAISS index (hnsw / ivf_flat / ivfpq)
├── evaluate_baseline.py # FAISS retrieval → Recall@K + latency
├── evaluate_reranked.py # 2‑stage FAISS → GPT‑4o reranker
├── encoder_clip.py # CLIP text encoder wrapper
├── encoder_sbert.py # SBERT text encoder wrapper
├── encoder_mmembed.py # MM‑Embed text encoder wrapper
├── retriever_faiss.py # minimal Flask/FastAPI retrieval service
├── Cost metrics.ipynb # notebook: parses logs, builds tables & plots
├── pipeline.png # high‑level pipeline diagram
├── recallvcost.png # recall vs. cost & scale plot
├── requirements.txt # pip dependencies
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
- [Conclusion](#Conclusion) 

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

Load up to N COCO captions, ```captions_val2017.json```,  encodes with your selected encoder, embed them in batches, and save to a ```.npy```
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

Builds and saves a FAISS index:

``` bash
python index_builder.py \
  --embeds data/embeds_val2017.npy \
  --out    models/hnsw_val2017.faiss \
  --engine hnsw        # or `ivf_flat`, `ivfpq`
  [--ef_search 64]     # for HNSW
  [--nlist 512 --pq_m 32]  # for IVF‑PQ
```
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

Recall slowly degrades as you process more candidates (```limit```) or ask the reranker to re‑score more (```top_k```), but cost is roughly linear. Sweet spot around 600-1500: you get ≈ 0.88–0.90 recall at sub‑$2 cost, with ~ 560 - 590 ms per query.

![Recall vs Cost](recallvcost.png)



### Pipeline Comparison @ 5 000 captions, top_k = 1


|               Pipeline              | Recall\@1 | Latency (ms/q) | Cost (US\$) |
| :---------------------------------: | :-------: | :------------: | :---------: |
|         CLIP + HNSW (ef=64)         |   1.0000  |      65.2      |     0.72    |
|       MM‑Embed + HNSW (ef=64)       |   0.9974  |      75.7      |     0.72    |
| MM‑Embed + IVF‑PQ (nlist=512, m=32) |   0.9958  |      70.2      |     0.72    |

CLIP+HNSW recovers every caption exactly (Recall@1 = 1.0) and is fastest (~ 65 ms). Swapping in MM‑Embed (a larger, 1 024‑dim vector) costs ~ 10 ms extra per query and drops recall by ~ 0.2–0.4 pp. Using IVF‑PQ instead of HNSW on the MM‑Embed vectors regains some speed (~ 70 ms) at only a tiny extra recall loss (~ 0.16 pp vs. HNSW).

![Pipeline](pipeline.png)


## What the numbers mean
Here is a summary what the numbers above mean: 
-Perfect, lightning‑fast retrieval with CLIP + HNSW: we recover every caption (Recall@1 = 1.0) in ~ 65 ms/query.

- Richer, higher‑dim features (MM‑Embed) still deliver > 99.5 % recall at ≈ 70–75 ms/query—at negligible extra cost.

- IVF‑PQ gives a small speed/memory win over HNSW on the 1 024‑D vectors, losing < 0.2 pp recall.

- Reranker experiments (on top‑K = 1, 5, 10…) show recall degrades gradually as you scale up how many candidates you rescore, with cost scaling linearly. Sweet spot around 600–1 500 candidates for ~ 0.88–0.90 recall at sub‑$2 cost (≈ 560–590 ms/query).




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

## Conclusion
Here we embed COCO captions into vectors (using CLIP‑Vision or MM‑Embed). Then we indexed those vectors with FAISS (HNSW or IVF‑PQ) for evaluating retrieval quality (Recall@K) and speed (ms/query). Finally, We rerank top candidates with GPT‑4o‑vision ("gpt‑4o‑instruct"), using real API credits.

