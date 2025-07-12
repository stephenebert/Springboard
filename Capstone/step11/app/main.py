# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      # 1) Grab the latest code
      - uses: actions/checkout@v3

      # 2) Install Python
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      # 3) Install your app + test deps
      - name: Install dependencies
        working-directory: Capstone/step11
        run: |
          pip install -r requirements.txt pytest requests

      # 4) Launch FastAPI (in the background), pointing at the tiny fixtures
      - name: Launch FastAPI (background)
        working-directory: Capstone/step11
        env:
          FAISS_INDEX_PATH: tests/fixtures/data_small/ivf_flat_small.index
          META_PATH:        tests/fixtures/data_small/id2meta_small.json
        run: |
          uvicorn app.main:app --host 0.0.0.0 --port 8000 & echo $! > server.pid

      # 5) Give it a moment to come up (or poll /health here)
      - name: Wait for server to be healthy
        working-directory: Capstone/step11
        run: sleep 2

      # 6) Run your pytest suite
      - name: Run pytest
        working-directory: Capstone/step11
        run: pytest -q tests
