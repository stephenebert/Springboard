import requests, numpy as np
BASE = "http://localhost:8000"

def random_unit_vec(d=512):
    x = np.random.randn(d).astype("float32")
    x /= np.linalg.norm(x)
    return x.tolist()

def test_search_returns_k():
    k = 3
    payload = {"query_vec": random_unit_vec(), "k": k}
    resp = requests.post(f"{BASE}/search", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()) == k
