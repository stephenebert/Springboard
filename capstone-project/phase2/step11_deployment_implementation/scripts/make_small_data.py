# scripts/make_small_data.py

import json
import numpy as np
import faiss
import h5py
from pathlib import Path

# 1) Point to real, full HDF5 embedding file + metadata

ROOT      = Path(__file__).parents[1]             # …/capstone-local
H5_FILE   = ROOT / "data" / "embeddings_full.h5"  # your full HDF5
META_FILE = ROOT / "data" / "metadata.parquet"    # or id2meta.json if you have it

# if only have metadata.parquet, convert to JSON first:
#   python - <<EOF
#   import pandas as pd, json
#   df = pd.read_parquet("data/metadata.parquet")
#   df = df[df.split=="train"]
#   df = df[["img_path","caption"]].to_dict(orient="records")
#   json.dump(df, open("data/id2meta.json","w"), indent=2)
#   EOF
# then set META_FILE = ROOT/"data"/"id2meta.json"

# 2) Where to write small subset
OUT_DIR = ROOT / "data_small"
OUT_DIR.mkdir(exist_ok=True)

N = 1000  # how many samples 

# 3) Extract first N embeddings from HDF5

with h5py.File(H5_FILE, "r") as f:
    embs = f["image_embeddings"][:N]          # shape (N, 512)
embs = embs.astype("float32")
np.save(OUT_DIR / "img_embs_small.npy", embs)

# 4) Slice metadata to the same N
# Here we assume there exists a pre-made id2meta.json next to the parquet
meta_json = ROOT / "data" / "id2meta.json"
if not meta_json.exists():
    raise FileNotFoundError(f"{meta_json} not found; please create it from the parquet")
with open(meta_json, "r", encoding="utf-8") as f:
    meta = json.load(f)
small_meta = meta[:N]
with open(OUT_DIR / "id2meta_small.json", "w", encoding="utf-8") as f:
    json.dump(small_meta, f, indent=2)

# 5) Build a tiny FAISS index over those N vectors

faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, str(OUT_DIR / "ivf_flat_small.index"))

print("Wrote small subset to", OUT_DIR)
