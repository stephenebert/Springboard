# Step 11: Deployment Implementation

[![Build Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/your-handle/capstone-retrieval)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docs.docker.com/get-docker/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A high-performance image retrieval system built with FAISS, FastAPI, and Docker. This system enables semantic search across large image datasets using CLIP embeddings with efficient indexing and real-time querying capabilities.

## Quick Start

### Prerequisites

- Docker Desktop with WSL 2 (Windows) or Docker Engine (Linux/Mac)
- Git

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-handle>/<name>
   cd <name>
2. **Build and start services**
   docker compose up -d --build
   
   - Initial build may take ~5 minutes to fetch dependencies
3. **Build FAISS index** (one-time setup)
  docker compose run --rm app \
    python scripts/build_index_step8.py \
      --h5   /app/step8/experiments/full/embeddings_full.h5 \
      --meta /app/step8/data/metadata.parquet \
      --split train \
      --out_dir /data/faiss-indexes
4. **Verify installation**
   curl http://localhost:8000/health
   
   \# Expected response: {"status":"ok", ...}

6. **Run smoke test**
   ## Services & Ports

| Service             | URL                       | Description         |
| :------------------ | :------------------------ | :------------------ |
| FastAPI Application | http://localhost:8000     | Main retrieval API  |
| Prometheus Metrics  | http://localhost:9090     | Monitoring dashboard |

## Project Structure

- `step-11/`
  - `app/`
    - `__init__.py`
    - `main.py` # FastAPI service
  - `data/`
    - `faiss-indexes/`
      - `ivf_flat_1024.index` # FAISS index file
    - `metadata_train.json` # Metadata mapping
    - `embeddings_full.h5` # HDF5 embeddings
    - `metadata.parquet` # Dataset metadata
  - `infra/`
    - `prometheus.yml` # Monitoring configuration
  - `scripts/`
    - `build_index_step8.py` # Index construction
    - `convert_jsonl.py` # Data conversion utilities
    - `scale_pipeline_hdf5.py` # Embedding pipeline
    - `smoke_test.py` # Integration testing
  - `tests/`
    - `test_health.py` # Health endpoint tests
    - `test_search.py` # Search functionality tests
  - `Dockerfile` # Container configuration
  - `docker-compose.yml` # Multi-service orchestration
  - `requirements.txt` # Python dependencies
  - `README.md`
 ## Key Components

 
## Core Scripts

| Script                   | Purpose                                                              | Usage                                      |
| :----------------------- | :------------------------------------------------------------------- | :----------------------------------------- |
| `scale_pipeline_hdf5.py` | Generate embeddings for full corpus (COCO + Flickr + StableDiff)     | `python scale_pipeline_hdf5.py --parquet data/metadata.parquet` |
| `build_index_step8.py`   | Build FAISS index from embeddings                                    | See installation step 3                    |
| `convert_jsonl.py`       | Convert JSONL to compact JSON format                                 | `python scripts/convert_jsonl.py`          |
| `smoke_test.py`          | End-to-end system validation                                         | `python smoke_test.py`                     |

## API Endpoints

* **GET** `/health` - Service health check
* **POST** `/search` - Semantic search endpoint
* **GET** `/metrics` - Prometheus metrics

## Usage Examples

### Basic Search Query

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cat sitting on a window",
    "top_k": 5
  }'
```

### Health Check
``` bash
curl http://localhost:8000/health
```

### Monitoring & Metrics

The system includes comprehensive monitoring through Prometheus:

* **Request latency** - `request_latency_seconds`
* **Request count** - `requests_total`
* **System health** - Available at `/metrics` endpoint

Access the Prometheus dashboard at [http://localhost:9090](http://localhost:9090) to visualize metrics.

### Testing

Run the test suite:

```bash
# Unit tests
pytest tests/

# Integration test
python smoke_test.py
```

