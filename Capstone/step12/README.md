# Step 12: Image-to-Text Retrieval Service

A production-ready, cross-modal retrieval API (FastAPI + FAISS) with a Gradio front-end. You can enter a text caption, hit **Submit**, and see the top-K nearest images rendered in a gallery.

---

## Checklist

| Criterion                          | How It’s Met                                                                                                                                                                                                                |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Correctness**                    | • FAISS IVF-Flat index loaded on startup, cosine similarity search in `/search` endpoint<br> Health check at `/health` returns index stats and liveness                                                                    |
| **Code Quality**                   | • PEP8-compliant Python modules<br> Clear separation: `app/main.py`, `app/routers`, `app/models`<br> Docstrings + type annotations                                                                                            |
| **Testing**                        | • Pytest fixtures under `tests/fixtures/data_small` with a mini index<br> `tests/test_health.py` and `tests/test_search.py` cover edge cases and end-to-end logic<br>• `docker-compose.test.yml` for isolated test runs      |
| **Documentation**                  | • Comprehensive **README** (this file)<br> In-code comments and module docstrings<br> Example `curl` commands                                                                                                              |
| **Deployment**                     | • `Dockerfile` for the API with layer caching<br> Auto-deploy on Render.com (Fargate)<br> Gradio app in `app.py` deployed on Hugging Face Spaces                                                                       |
| **Demonstration & UX**             | • Live **API**: `https://capstone-retrieval-api.onrender.com`<br> Live **UI**: `https://huggingface.co/spaces/<your-user>/retrieval-demo`<br>•Images rendered in a gallery, not raw JSON; friendly slider for k             |

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
