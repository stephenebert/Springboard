"""
smoke_test.py one-shot sanity-check for a local retrieval API.

Run from the repo root:

    python smoke_test.py
"""

import json, textwrap, h5py, numpy as np, requests, pathlib, sys

DATA_DIR = pathlib.Path("data")
H5_PATH  = DATA_DIR / "embeddings_full.h5"
API      = "http://localhost:8000"

# 1) Grab an embedding
if not H5_PATH.exists():
    sys.exit(f"{H5_PATH} not found — is data/ mounted?")

with h5py.File(H5_PATH, "r") as h5:
    # find the text-embedding dataset automatically
    key = next(k for k in h5 if "text" in k.lower())   # e.g. /text_embeddings
    vec = h5[key][0]                                   # first 512-d vector

#2) Query the service
resp = requests.post(
    f"{API}/search",
    json={"query_vec": vec.tolist(), "k": 3},
    timeout=15,
)

print(f"\n -->  status  {resp.status_code}")
print(f"-->  headers {resp.headers.get('content-type','')}")
print("--> first 300 bytes of body:\n",
      textwrap.indent(resp.text[:300], "   "),
      "\n")

# Pretty-print JSON only when response *is* JSON
if resp.headers.get("content-type","").startswith("application/json"):
    print("--> parsed JSON:\n",
          textwrap.indent(json.dumps(resp.json(), indent=2)[:800], "   "))
else:
    print("Body is not JSON:see docker logs for details.")
