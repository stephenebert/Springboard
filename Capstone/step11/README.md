# Step 11: Deployment Implementation

> **Status:** *in progress*
> This README evolves as we knock out the remaining rubric items. Everything below already works on a laptop.

---

## TL;DR – How to run it now

```bash
# 0️.  prerequisites
#    - Docker Desktop + WSL 2 (Windows) / Docker Engine (Linux/Mac)
#    - git clone https://github.com/<your‑handle>/capstone-retrieval
cd capstone-retrieval

# 1️.  Build + start services (FastAPI, Prometheus, DynamoDB‑local)
docker compose up -d --build   # first run fetches deps, ~5 min on laptop

# 2️.  Build FAISS index *once* (≈3 min on CPU)
docker compose run --rm app \
  python scripts/build_index_step8.py \
    --h5   /app/step8/experiments/full/embeddings_full.h5 \
    --meta /app/step8/data/metadata.parquet \
    --split train \
    --out_dir /data/faiss-indexes  

# 3️.  Check liveness
curl http://localhost:8000/health   # --> {"status":"ok", ...}

# 4️.  Quick smoke test
python smoke_test.py                # prints top‑K JSON
```

### Ports

| Service     | URL                                            | Note                        |
| ----------- | ---------------------------------------------- | --------------------------- |
| FastAPI app | [http://localhost:8000](http://localhost:8000) |                             |
| Prometheus  | [http://localhost:9090](http://localhost:9090) | scrapes `/metrics` from app |

---

## Project Map (field‑guide)

| File / dir                     | Purpose                                                                           | Typical command                                                 |
| ------------------------------ | --------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `scale_pipeline_hdf5.py`       | **Embed** full corpus (COCO + Flickr + StableDiff) → `embeddings_full.h5`         | `python scale_pipeline_hdf5.py --parquet data/metadata.parquet` |
| `scripts/build_index_step8.py` | **Index** image embeddings → `ivf_flat_1024.index` + `metadata_train.json`        | *see step 2 above*                                              |
| `scripts/convert_jsonl.py`     | Convert line‑delimited JSONL → compact JSON (only if needed)                      | `python scripts/convert_jsonl.py`                               |
| `app/main.py`                  | FastAPI micro‑service. Loads FAISS index, serves `/health`, `/search`, `/metrics` | launched automatically by `uvicorn` inside container            |
| `smoke_test.py`                | Posts first text‑embedding row to `/search` and pretty‑prints response            | `python smoke_test.py`                                          |
| `Dockerfile`                   | Builds `retrieval_app` image                                                      | `docker compose build app`                                      |
| `docker-compose.yml`           | Orchestrates app + Prometheus + DynamoDB‑local                                    | `docker compose up -d`                                          |
| `infra/prometheus.yml`         | Scrape config (every 15 s)                                                        | auto‑loaded                                                     |

---

## Repository Tree

```text
capstone-local/
├── app/
│   ├── __init__.py
│   └── main.py                    # FastAPI service (loads FAISS index)
│
├── data/                          # Data is stored locally due to being too large
│   ├── faiss-indexes/
│   │   ├── ivf_flat_1024.index    # built by build_index_step8.py
│   │   └── metadata_train.json    # converted by convert_jsonl.py
│   ├── embeddings_full.h5         # full HDF5 embeddings (step 8)
│   ├── img_embs_full.npy
│   ├── txt_embs_full.npy
│   └── metadata.parquet
│
├── infra/
│   └── prometheus.yml             # scrape config
│
├── scripts/
│   ├── build_index_step8.py
│   ├── convert_jsonl.py
│   ├── scale_pipeline_hdf5.py
│   └── smoke_test.py
│
├── tests/
│   ├── test_health.py             # unit test for /health
│   └── test_search.py             # unit test for /search
│
├── Dockerfile                     # builds the app image
├── docker-compose.yml             # full stack (FastAPI + DynamoDB + Prometheus)
├── requirements.txt               # Python deps for the image
└── README.md
```                 
---


### How to add it

1. Open `README.md`.
2. Find the spot right after the "Project Map" table
3. Paste the block above.
4. Save & commit:

```bash
git add README.md
git commit -m "docs: add repo tree to README"
git push origin main
```


## Roadmap to Step 11

* [x] Local Compose stack with FastAPI + FAISS + Prometheus
* [x] Build large IVF‑Flat index (`nlist=1024`, `nprobe=16`)
* [x] Smoke test returns 200 + top‑K JSON
* [x] Unit tests (`pytest`)
* [ ] CI workflow (GitHub Actions) – build, test, push image artifact
* [ ] Prometheus metrics → Grafana Cloud (free tier) dashboard
* [ ] Terraform lean Fargate stack (Spot + HTTP API) with budget guardrails

---

## Development Tips

* **Re‑index quickly** – If one tweaks `nlist`, run `build_index_step8.py` again; it streams HDF5 in \~3 min on CPU.
* **RAM usage** – IVF‑Flat index (\~1.7 GB) fits in a 2 GB container; adjust `mem_limit` in `docker-compose.yml` if needed.
* **p95 latency** – Prometheus scrapes `/metrics`; check `<http://localhost:9090>` ▸ *Graph* ▸ `request_latency_seconds`.

---

## Licensing & Credits

* CLIP ViT‑B/32 weights © OpenAI, licensed MIT.
* FAISS © Facebook AI Research, BSD‑2‑Clause.
* Dataset captions from COCO, Flickr30k, Stable Diffusion prompts (CC‑BY).

---
