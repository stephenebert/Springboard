"""
smoke_test_small.py
-------------------
One-shot sanity check against the *mini-dataset* stack.

Assumptions
-----------
* docker-compose.test.yml maps the FastAPI container to http://localhost:8010
* tests/fixtures/data_small/ contains:
      img_embs_small.npy    - Nx512 float32 vectors
      ivf_flat_small.index  - FAISS index   (loaded by the app)
      id2meta_small.json    - metadata      (loaded by the app)

Run from the repo root:

    python scripts/smoke_test_small.py
"""

import json
import pathlib
import sys
import textwrap

import numpy as np
import requests

# Config: override via env vars if the user wants
FIXTURE_DIR = pathlib.Path("tests/fixtures/data_small")
NPY_PATH    = FIXTURE_DIR / "img_embs_small.npy"
API_ROOT    = "http://localhost:8010"      # matches docker-compose.test.yml
K_NEIGHBORS = 2                            # tiny dataset

# 1) Load a sample 512-D embedding
if not NPY_PATH.exists():
    sys.exit(f"[ERROR] {NPY_PATH} not found – did you clone the fixtures?")

vec = np.load(NPY_PATH)[0]                # (512,) float32

# 2) POST /search
try:
    resp = requests.post(
        f"{API_ROOT}/search",
        json={"query_vec": vec.tolist(), "k": K_NEIGHBORS},
        timeout=10,
    )
except requests.exceptions.RequestException as e:
    sys.exit(f"[ERROR] request failed: {e}")

# 3) Pretty-print the reply
print(f"\n--> status  {resp.status_code}")
print(f"--> headers {resp.headers.get('content-type', '')}")
print("--> first 300 bytes of body:\n",
      textwrap.indent(resp.text[:300], "   "), "\n")

if resp.headers.get("content-type", "").startswith("application/json"):
    print("--> parsed JSON:\n",
          textwrap.indent(json.dumps(resp.json(), indent=2)[:800], "   "))
else:
    print("Body isn’t JSON; check docker logs for details.")
