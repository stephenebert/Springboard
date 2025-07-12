import requests
BASE = "http://localhost:8000"

def test_health_endpoint():
    resp = requests.get(f"{BASE}/health")
    assert resp.status_code == 200               # <- must be alive
    data = resp.json()
    assert "index_dim"  in data
    assert "index_size" in data
