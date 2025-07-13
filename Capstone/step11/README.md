# Step 11 · Deployment Implementation 


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

