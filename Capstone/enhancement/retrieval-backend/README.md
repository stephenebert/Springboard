
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
