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


# 0. Prerequisites 
To run the project exactly as described (locally or in CI) you only need three things:
| Component                                | Windows                                                | macOS / Linux                         | Why it’s needed                                                                            |
| ---------------------------------------- | ------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Docker** (with Compose v2)             | **Docker Desktop** + **WSL 2** backend (Windows 11/10) | **Docker Engine** (or Docker Desktop) | Runs the FastAPI container, Prometheus, DynamoDB-local, etc.                               |
| **Git**                                  | Git for Windows (or any Git client)                    | Git (Homebrew / apt / pacman)         | Clone the repo & pull updates.                                                             |
| **Python 3.8 – 3.11** (optional on host) | `pyenv` / Windows Store / Anaconda                     | System Python / `pyenv` / Homebrew    | Only needed to launch the local smoke-test scripts; the service itself runs inside Docker. |

Minimal hardware:

1. ≥ 8 GB RAM (4 GB free): full IVF-Flat index (~1.7 GB RAM) plus containers.

2. ≈ 4 GB disk for Docker image layers + mini fixtures (full data/ folder stays local and is ignored by Git).

3. Internet access the first time you pull Docker base images.

- Note: If you’re just running the mini-fixture stack (port 8010) or running CI, < 2 GB RAM is sufficient because the tiny FAISS index is ≈ 2 MB.


# 1. Build + launch full stack  (FastAPI 8000, DynamoDB-local, Prometheus)
``` bash
# From the repo root
docker compose up -d --build   # first run ≈ 5 min on a laptop
```
What happens:
| Phase                | What Docker does                                                                                               | Notes                                                                                       |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Build image**      | Uses the `Dockerfile`, installs Python + requirements, copies `app/` code.                                     | Final image ≈ 650 MB. On subsequent runs Docker re-uses cached layers, so it’s much faster. |
| **Create network**   | `capstone-local_default` virtual bridge for inter-container traffic.                                           |                                                                                             |
| **Start containers** | 1. `retrieval_app` (FastAPI)<br>2. `dynamodb_local` (local metadata store)<br>3. `prometheus` (metrics scrape) | `-d` runs them in the background (detached mode).                                           |

Verify everything came up:

``` bash
docker compose ps
```
Expected output (example):
``` bash
NAME               IMAGE                    STATUS           PORTS
retrieval_app      capstone-local-app       Up 0.0.0.0:8000->8000/tcp
dynamodb_local     amazon/dynamodb-local    Up 0.0.0.0:8001->8000/tcp
prometheus         prom/prometheus:v2.52.0  Up 0.0.0.0:9090->9090/tcp
```
Troubleshooting
- If ports are already in use (error "Bind for 0.0.0.0:8000 failed, port is already allocated"):
``` bash
docker compose down   # stop previous stack
sudo lsof -i :8000    # find the process using the port (Linux/macOS)
netstat -ano | findstr :8000  # Windows
```
-If ``` retrieval_app`` exits with code 137, your machine is low on RAM then allocate ≥ 4 GB to Docker Desktop.

Once you see STATUS Up for all three containers, move on to Step 2 (liveness check).

# 2. Check liveness (```/health```)
After the containers are Up, confirm the FastAPI service is responsive.

``` bash
# Linux / macOS
curl http://localhost:8000/health

# Windows PowerShell
Invoke-WebRequest http://localhost:8000/health | Select-Object -Expand Content
```

Expected JSON reply (example):



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
