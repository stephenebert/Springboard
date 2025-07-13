import requests

BASE = "http://localhost:8000"

def test_health_endpoint():
    resp = requests.get(f"{BASE}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_dim"] == 512          # index is 512-D
    assert data["index_size"] > 0            # something is in the index
