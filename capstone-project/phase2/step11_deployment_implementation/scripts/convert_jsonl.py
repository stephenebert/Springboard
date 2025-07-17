#!/usr/bin/env python3
import json
from pathlib import Path

p = Path("data/faiss-indexes/metadata_train.jsonl")
out = [json.loads(line) for line in p.read_text().splitlines()]
# write out as a single JSON array that app/main.py can consume
Path("data/id2meta.json").write_text(json.dumps(out))
print("Wrote data/id2meta.json")
