# Step 11: Deployment Implementation

![Status](https://img.shields.io/badge/status-complete-brightgreen)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docs.docker.com/get-docker/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Cross-modal **image-to-text retrieval** micro-service.  
Dockerised with FastAPI + FAISS, orchestrated locally via **docker-compose**.

Two datasets:
- **Full** corpus: COCO + Flickr + Stable Diffusion (>850k rows, runs on port 8000)  
  *lives only on the user's laptop – not committed*
- **Mini** fixture: 1,000 rows for CI (runs on port 8010)

---

## High-Level Overview

| Milestone                              | What we delivered                                                                                                             | Importance                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Local micro-service**                | FastAPI + FAISS image-to-text retrieval app (`app/main.py`) wrapped in a Docker image.                                        | Same container later publish to ECR, Fargate, etc.                              |
| **Offline artifacts**                  | Full `embeddings_full.h5`, `ivf_flat_1024.index`, `metadata_train.json` built via `build_index_step8.py`.                 | Keeps inference fast (<150 ms p95) by loading a pre-trained IVF-Flat index.     |
| **Mini fixture**                       | `tests/fixtures/data_small/` (1,000 vectors + 2MB index).                                                                    | Lets CI run <2 min.                                                              |
| **docker-compose stacks**              | *Full* stack (8000) + *Test* override (8010) with Prometheus and DynamoDB-local.                                              | One-liner startup; mirrors future cloud topology.                                |
| **Health, search & metrics endpoints** | `/health`, `/search`, `/metrics` (via `prometheus_fastapi_instrumentator`).                                                  | Observability & SLO verification.                                                |
| **Smoke tests**                        | `smoke_test.py` (full) and `smoke_test_small.py` (mini).                                                                      | One-shot sanity check; used by CI.                                               |
| **Unit / integration tests**           | `tests/test_health.py`, `tests/test_search.py` --> 2 / 2 pass.                                                                | Guarantees basic behaviour before each deploy.                                   |
| **GitHub Actions CI** (`ci.yml`)       | Installs deps --> starts FastAPI --> runs pytest --> builds Docker image --> (optional push).                                 | Proves image + tests + Dockerfile work cleanly.                                  |
| **Repo hygiene**                       | Global `.gitignore` excludes large data; small fixture whitelisted.                                                          | Keeps repo lean (<10MB) while data stays local.                                  |
| **Documentation**                      | Expanded README with run guides, smoke-test instructions, badges.                                                            | Reproducible by others.                                                          |

---

## Project Structure

```
step-11/
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── faiss-indexes/
│   │   └── ivf_flat_1024.index
│   ├── embeddings_full.h5
│   ├── metadata_train.json
│   └── metadata.parquet
├── infra/
│   └── prometheus.yml
├── scripts/
│   ├── build_index_step8.py
│   ├── convert_jsonl.py
│   ├── scale_pipeline_hdf5.py
│   └── smoke_test.py
├── tests/
│   ├── test_health.py
│   └── test_search.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 0. Prerequisites

| Component                     | Windows                            | macOS / Linux                     | Why it’s needed                                    |
|------------------------------|-------------------------------------|----------------------------------|---------------------------------------------------|
| Docker (Compose v2)          | Docker Desktop + WSL 2             | Docker Engine or Desktop         | Run containers                                    |
| Git                          | Git for Windows                    | Git (brew / apt)                 | Clone repo, version control                       |
| Python 3.8 – 3.11 (optional) | Anaconda / Windows Store / pyenv   | System Python / Homebrew         | For running local smoke tests                    |

Hardware: ≥8 GB RAM for full index; <2 GB for mini-fixture.

---

## 1. Build + Launch Full Stack

```bash
docker compose up -d --build
```

To verify:
```bash
docker compose ps
```

If port errors:
```bash
docker compose down
sudo lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

---

## 2. Health Check

```bash
curl http://localhost:8000/health
```
Expected JSON:
```json
{
  "status": "ok",
  "index_dim": 512,
  "nprobe": 16,
  "index_size": 850668
}
```

Use Swagger at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 3. Smoke Test

```bash
python scripts/smoke_test.py
```
Expected:
- HTTP 200
- JSON with `results`
- Score between 0–1

For mini fixture:
```bash
python scripts/smoke_test_small.py
```

---

## Mini-Fixture Stack (CI)
```bash
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d
curl http://localhost:8010/health
python scripts/smoke_test_small.py
pytest -q tests
```

---

## Path Map

| Path                                | Purpose                                     |
|-------------------------------------|---------------------------------------------|
| app/main.py                         | FastAPI app                                 |
| scripts/build_index_step8.py        | FAISS index builder                         |
| scripts/smoke_test.py               | Full smoke test                             |
| scripts/smoke_test_small.py         | Mini-fixture test                           |
| tests/test_*.py                     | CI unit tests                               |
| docker-compose.yml                  | Full stack                                  |
| docker-compose.test.yml             | Override for mini                           |
| infra/prometheus.yml                | Metrics scrape config                       |
| .github/workflows/ci.yml            | GitHub Actions CI config                    |

---

## Checklist

- [x] Dockerised FastAPI service
- [x] Precomputed IVF-Flat index
- [x] Local stack with monitoring sidecars
- [x] `/health`, `/search`, `/metrics` endpoints
- [x] Smoke + unit tests
- [x] CI workflow (green)
- [x] Full documentation

---

## Tips & Credits

- Reindex: tweak `nlist`, rerun `build_index_step8.py`
- Port conflicts: full = 8000, mini = 8010
- Prometheus + Grafana integration possible
- FAISS: © Meta (BSD); CLIP: © OpenAI (MIT); Datasets: CC-BY

You're now running a fast, CI-backed, dockerized image-to-text retrieval service!
