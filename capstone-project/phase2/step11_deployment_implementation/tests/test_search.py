import numpy as np, requests, math

BASE = "http://localhost:8000"

def random_unit_vec(dim: int = 512) -> list[float]:
    v = np.random.randn(dim).astype("float32")
    v /= math.sqrt((v ** 2).sum())           # L2-normalize for cosine sim
    return v.tolist()

def test_search_returns_k():
    k = 3
    payload = {"query_vec": random_unit_vec(), "k": k}
    resp = requests.post(f"{BASE}/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == k
    for hit in data["results"]:
        # minimal sanity checks on each hit
        assert {"id", "image_path", "caption", "score"} <= hit.keys()
