# Step 11 · Deployment Implementation 
[![Build Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/your-handle/capstone-retrieval)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docs.docker.com/get-docker/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Cross-modal **image-to-text retrieval** micro-service.  
Dockerised with FastAPI + FAISS, orchestrated locally via **docker-compose**.  
Two datasets:

* **Full** corpus: COCO + Flickr + Stable Diffusion (>850 k rows, runs on port 8000)  
  *lives only on your laptop – not committed*
* **Mini** fixture: 1 000 rows for CI & grading (runs on port 8010)

Everything below already works on a fresh clone.

---

## Run it right now 


# 0. prereqs
#    Docker Desktop + WSL 2 (Win) / Docker Engine (Linux/Mac)
git clone https://github.com/<YOUR-GITHUB-HANDLE>/<REPO-NAME>.git
cd <REPO-NAME>

# 1. Build + launch full stack  (FastAPI 8000, DynamoDB-local, Prometheus)
docker compose up -d --build

# 2. Check liveness
curl http://localhost:8000/health

# 3. Smoke test (large index)
python scripts/smoke_test.py

## Mini-fixture stack (CI / quick demo)

``` bash
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d
curl http://localhost:8010/health
python scripts/smoke_test_small.py          # hits port 8010
pytest -q tests                             # 2 tests, < 1 s
docker compose -f docker-compose.yml -f docker-compose.test.yml down
```
| Service        | URL / port                                     | Note                              |
| -------------- | ---------------------------------------------- | --------------------------------- |
| FastAPI (full) | [http://localhost:8000](http://localhost:8000) | `/health`, `/search`, `/metrics`  |
| FastAPI (mini) | [http://localhost:8010](http://localhost:8010) | override in *docker-compose.test* |
| Prometheus     | [http://localhost:9090](http://localhost:9090) | scrapes FastAPI metrics           |

## Project Map

| Path / File                               | Purpose                                                                            | Typical command                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| `app/main.py`                             | FastAPI micro-service, loads FAISS index, exposes `/health`, `/search`, `/metrics` | auto-started by `uvicorn` in container    |
| `scripts/build_index_step8.py`            | Build full **IVF-Flat** index (`ivf_flat_1024.index`, `metadata_train.json`)       | `python ... --h5 data/embeddings_full.h5` |
| `scripts/smoke_test.py`                   | Posts first text vector to `/search` (port 8000)                                   | `python scripts/smoke_test.py`            |
| `scripts/smoke_test_small.py`             | Same, but targets mini stack (port 8010)                                           | `python scripts/smoke_test_small.py`      |
| `tests/fixtures/data_small/`              | Tiny dataset → **committed** (N = 1 000)                                           | loaded automatically by tests             |
| `tests/test_health.py` / `test_search.py` | CI / grading unit tests                                                            | `pytest -q tests`                         |
| `docker-compose.yml`                      | Full stack (FastAPI 8000 + DynamoDB + Prom)                                        | `docker compose up -d`                    |
| `docker-compose.test.yml`                 | Override: mounts mini data, maps 8010                                              | `docker compose -f ... up -d`             |
| `infra/prometheus.yml`                    | Minimal scrape config (15 s)                                                       | auto-mounted                              |
| `.github/workflows/ci.yml`                | GitHub Actions: install deps → run tests → build image                             | triggers on push / PR                     |

## Checklist 
 1. Dockerised FastAPI service

 2. IVF-Flat index pre-computed offline (nlist = 1024, nprobe configurable)

 3. Local compose stack with Prometheus & DynamoDB-local side-cars

 4. /health & /search endpoints + p95 latency metrics

 5. Smoke tests (full & mini)

 6. Unit / integration tests pass in CI (mini fixture)

 7. GitHub Actions workflow green

 8. Documentation (this README)
## Development Tips
1. Re-index quickly: tweak nlist / nprobe, re-run build_index_step8.py (approximately 3 min on CPU).

2. Port collisions: full uses 8000, mini uses 8010 to avoid conflicts.

3. Monitoring – Prometheus scrapes /metrics; connect Grafana later.

4. Costs: local only. Cloud Fargate prototype (Step 12) is approximately $6/mo.

## Licensing & Credits
1. CLIP ViT-B/32 weights © OpenAI (MIT).
2. FAISS © Meta AI Research (BSD-2-Clause).
3. Dataset captions: COCO, Flickr30k, Stable Diffusion prompts (CC-BY).
4. 
