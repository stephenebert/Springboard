## 🗄️ Building the COCO Caption FAISS Assets Yourself

> **Skip this if you downloaded the ready-made files via
> `python scripts/get_assets.py`.**  
> The steps below let you rebuild the 600 k-vector FAISS index from scratch.

| Output file | Approx. size | Description |
|-------------|--------------|-------------|
| `scripts/coco_caption_clip.index` | **1.2 GB** | FAISS `IndexFlatL2` with 591 753 × 512 float32 vectors |
| `scripts/coco_caption_texts.npy` | **0.6 GB** | NumPy `object` array containing the same captions in the index order |

---

### 1. Choose how to ingest COCO captions

| Method | Pros | Cons |
|--------|------|------|
| **A. Hugging Face datasets** (recommended) | 1-liner download, no zip juggling, auto-caching | Adds `datasets` dependency (~100 MB install) |
| **B. Manual ZIP** (raw JSON) | Zero extra libs | A bit more typing |

---

#### **A — Using Hugging Face datasets (fast & easy)**

```bash
# activate your environment first
conda activate capstone-gradio-py310     # or your venv

pip install datasets tqdm  # if not already present
python scripts/build_index_from_hf.py

#!/usr/bin/env python
"""
Build coco_caption_clip.index & coco_caption_texts.npy from the
Hugging Face  'mscoco' caption dataset.
"""
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, tqdm, os, pathlib

OUT_DIR = pathlib.Path(__file__).resolve().parent   # scripts/
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "clip-ViT-B-32"
DEVICE     = "cpu"          # "mps" on Apple-silicon, "cuda" if you have a GPU
BATCH      = 1024           # adjust for RAM

print("Loading COCO captions via Hugging Face datasets …")
ds = load_dataset("mscoco", "2014")                # train+val 2014 split
texts = [rec["caption"] for rec in ds["train"]] + \
        [rec["caption"] for rec in ds["validation"]]
print(f"Loaded {len(texts):,} captions.")

print(f"Embedding with {MODEL_NAME} on {DEVICE} …")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
vecs = model.encode(
    texts,
    batch_size=BATCH,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
).astype("float32")

print("Building FAISS IndexFlatL2 …")
index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)

faiss_path = OUT_DIR / "coco_caption_clip.index"
capt_path  = OUT_DIR / "coco_caption_texts.npy"

print("Saving:")
faiss.write_index(index, str(faiss_path))
np.save(capt_path, np.array(texts, dtype=object))
print("Done.")
print(f"{faiss_path.name}  → {faiss_path.stat().st_size/1e9:.2f} GB")
print(f"{capt_path.name}   → {capt_path.stat().st_size/1e9:.2f} GB")
```
#### **B — Manual ZIP download**
