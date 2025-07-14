# Step 12: Image-to-Text Retrieval Service


A **production-ready**, cross-modal retrieval system that ties together:

1. **Efficient ANN Search API**  
   - FastAPI service loading a FAISS IVF-Flat index on startup  
   - High-throughput, low-latency cosine similarity search  
   - `/health` endpoint for liveness & index stats  
   - `/search` endpoint accepting 512-dim embeddings + `k` --> top-K results  

2. **Gradio Front-End**  
   - User-friendly UI to type a caption and choose **k** via slider  
   - Renders image thumbnails in a responsive gallery instead of raw JSON  
   - Hosted on Hugging Face Spaces, wired to call your live API  

3. **Robust Development & Deployment**  
   - Comprehensive pytest suite with fixtures & mini-index  
   - Dockerfile + `docker-compose.test.yml` for CI & local testing  
   - Auto-deploy on Render.com (Docker) and Hugging Face Spaces (Gradio)  

---

## Features

- **Correctness**  
  - Verified FAISS index load & ANN search semantics  
  - Pydantic schemas enforce 512-dim input vector & valid `k`  
- **Code Quality**  
  - Modular structure (`app/`, `routers/`, `models/`)  
  - Clear docstrings & type annotations  
- **Testing**  
  - End-to-end tests against a small "canary" index  
  - Health & search behavior for boundary and error cases  
- **Documentation**  
  - This README; in-code comments; curl examples  
- **Deployment**  
  - Dockerized API with multi-stage build caching  
  - Gradio Space with public URL binding via `API_URL` var  
---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Capstone.git
cd Capstone/step12
```

### 2. Prepare your data

By default the service looks for:

- ```FAISS_INDEX_PATH``` to ```/data/ivf_flat_1024.index```
- ```META_PATH → /data/id2meta.json```
You can override via environment variables or a ```.env``` file:

``` bash
export FAISS_INDEX_PATH=/path/to/ivf_flat_1024.index
export META_PATH=/path/to/id2meta.json
export NPROBE=16
```


### 3. Build & run the API
- Locally with Uvicorn
  
  ``` bash
  pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- With Docker

``` bash
docker build -t capstone-retrieval-api .
docker run -e FAISS_INDEX_PATH=/data/ivf_flat_1024.index \
           -e META_PATH=/data/id2meta.json \
           -p 8000:8000 \
           capstone-retrieval-api
```

### 4. Hit the endpoints
- Health check prints out ```{"status":"ok","index_dim":512,"nprobe":16,"index_size":1000}```
  ``` bash
  curl https://capstone-retrieval-api.onrender.com/health
 ```
- Search builds a 512-dim float list and POST:

``` bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"query_vec":[0.0,0.0,…512 floats total…], "k":3}' \
     https://capstone-retrieval-api.onrender.com/search
```
### 5. Run tests
``` bash
docker-compose -f docker-compose.test.yml up --build --exit-code-from tests
```
## Gradio Front-end
1. In the root of this repo, install demo deps:
   ``` bash
   pip install -r requirements.txt
   ```
2. Set your API URL in Settings -> Variables of your HF Space:
   ``` ini
   API_URL = https://capstone-retrieval-api.onrender.com
   ```
3. Launch locally
   ``` bash
   python app.py
   ```
4. Deploy by pushing ```app.py``` and ```requirements.txt``` to your HF Space.
Live demo at ```https://huggingface.co/spaces/<your-user>/retrieval-demo```

The HF app looks like this
![Render Deployment Logs](images/Screenshot%202025-07-14%20004518.png)
