# Capstone Project: Image-to-Text Cross-Modal Retrieval

**Author**: Stephen Ebert  
**Bootcamp**: Machine Learning Engineering - Springboard  
**Model Type**: Deep Learning + Cross-Modal Retrieval  
**Frontend**: Gradio  
**Backend**: FastAPI + FAISS  
**Deployment**: Docker, Hugging Face Spaces, Render.com  

---

## Overview

This project demonstrates a **cross-modal retrieval system**, where users can input a text query (caption) and retrieve the top-K matching images from a large-scale embedding database.  

The system is optimized for both **performance** (via FAISS ANN search) and **usability** (via a Gradio front-end).

---

## Repository Structure

```
capstone-project/
├── extra_exploration/
├── images/
├── phase1/
│   ├── step1_initial_project_ideas/
│   ├── step2_data_collection/
│   ├── step3_project_proposal/
│   ├── step4_survey_existing_research/
│   ├── step5_data_wrangling/
│   └── step6_benchmark_model/
├── phase2/
│   ├── step7_experiment_models/
│   ├── step8_scale_prototype/
│   ├── step9_deployment_method/
│   ├── step10_deployment_design/
│   ├── step11_deployment_implementation/
│   └── step12_share_project/
└── README.md
```

---

## Key Features

### Machine Learning Engineering
- **Deep Learning Feature Extractors** (CLIP via Hugging Face)
- **Vector Storage & Indexing**: Efficient cosine similarity search via FAISS IVF-Flat
- **Data Preprocessing Pipelines**: HDF5, JSONL, COCO-format parsing
- **Evaluation Metrics**: Top-K accuracy, cosine thresholds

### Software Architecture
- **FastAPI service** with `/health` and `/search` endpoints
- **Gradio front-end** for interactive text-to-image search
- **Modular codebase** with proper type hints and docstrings
- **Test suite** with unit + smoke tests
- **Monitoring** via Prometheus (optional)

### Deployment
- **Dockerized** service with `Dockerfile` and `docker-compose.yml`
- **CI-ready** with GitHub Actions
- **Public deployment** on:
  - [Render.com (FastAPI API)](https://capstone-retrieval-api.onrender.com)
  - [Hugging Face Spaces (Gradio UI)](https://huggingface.co/spaces/<your-username>/retrieval-demo)

---

## How It Works

1. A **text query** is converted into a 512-dim embedding via CLIP.
2. The query is passed to a **FastAPI server**, which runs a cosine similarity search against a pre-indexed FAISS database.
3. The top-K image IDs are returned with metadata and rendered by Gradio as thumbnails.

---

## Quick Start Guide

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/capstone.git
cd capstone/step12
```

### 2. Prepare your data

The service expects the following:
```bash
FAISS_INDEX_PATH=/data/ivf_flat_1024.index
META_PATH=/data/id2meta.json
NPROBE=16
```

You can export them or place in a `.env` file.

---

### 3. Run the FastAPI backend

#### Locally with Uvicorn:
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Or with Docker:
```bash
docker build -t capstone-retrieval-api .
docker run -e FAISS_INDEX_PATH=/data/ivf_flat_1024.index \
           -e META_PATH=/data/id2meta.json \
           -p 8000:8000 \
           capstone-retrieval-api
```

### 4. Check the API

#### Health check:
```bash
curl https://capstone-retrieval-api.onrender.com/health
```

Expected:
```json
{"status":"ok","index_dim":512,"nprobe":16,"index_size":1000}
```

#### Sample search query:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query_vec":[0.1, 0.2, ..., 512 floats], "k":3}' \
  https://capstone-retrieval-api.onrender.com/search
```

---

### 5. Run the Gradio front-end

```bash
pip install -r requirements.txt
python app.py
```

OR deploy to Hugging Face Spaces with:

```
app.py
requirements.txt
```

Set environment variable in HF Space:
```ini
API_URL=https://capstone-retrieval-api.onrender.com
```

---

## Testing & CI

- Smoke tests:
```bash
python scripts/smoke_test.py
```
- Unit tests:
```bash
pytest -q tests
```
- CI:
  - GitHub Actions runs `docker-compose.test.yml`, smoke tests, and Docker build

---

## Deployment Architecture

Deployed on two platforms:
1. **Render.com** hosts the FastAPI inference service (auto-redeploys from GitHub)
2. **Hugging Face Spaces** hosts the Gradio demo UI (calls the Render backend)

Dockerized end-to-end with reproducible environments and support for scaling.


---

## Skills Demonstrated

This project demonstrates full-stack ML engineering competencies:
- Project design and scoping
- Data collection, cleaning, wrangling
- Embedding models and similarity metrics
- Scalable vector search with FAISS
- FastAPI production services
- UI/UX with Gradio
- Docker & CI/CD deployment
- Monitoring and logging architecture

---

## 📁 Datasets

- **COCO Captions**
- **Flickr-30K**
- **Stable Diffusion Synthetic Images**

Image metadata stored in `id2meta.json`. Embeddings precomputed and indexed.

---

## Evaluation Metrics

- Top-K retrieval accuracy
- Cosine similarity thresholds
- Health check coverage
- Integration and smoke test success rate

---

## Screenshots

### Gradio Frontend
![Demo UI](images/Screenshot%202025-07-14%20004105.png)

### Render Logs
![Deployment Logs](images/Screenshot%202025-07-14%20004518.png)

---

## License

MIT License. FAISS (© Meta), CLIP (© OpenAI), Datasets (CC-BY).

---

## Acknowledgments

This project was completed as part of the Springboard Machine Learning Engineering bootcamp.  
Thanks to the mentors, reviewers, and instructors who supported this journey.

---

