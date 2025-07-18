# Building the COCO Caption FAISS Assets

> **Skip this if you downloaded the ready-made files via `python scripts/get_assets.py`.**  
> The steps below let you rebuild the 600k-vector FAISS index from scratch.

## Output Files

| Output file | Approx. size | Description |
|-------------|--------------|-------------|
| `scripts/coco_caption_clip.index` | **1.2 GB** | FAISS `IndexFlatL2` with 591,753 × 512 float32 vectors |
| `scripts/coco_caption_texts.npy` | **0.6 GB** | NumPy `object` array containing the same captions in index order |

## Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| **A. Hugging Face datasets** (recommended) | One-liner download, no zip juggling, auto-caching | Adds `datasets` dependency (~100 MB install) |
| **B. Manual ZIP** (raw JSON) | Zero extra dependencies | More manual steps |

## Method A: Using Hugging Face Datasets (Recommended)

### Prerequisites
```bash
# Activate your environment first
conda activate capstone-gradio-py310     # or your venv

# Install required packages
pip install datasets tqdm sentence-transformers faiss-cpu
```

### Build Script
Create `scripts/build_index_from_hf.py`:

```python
#!/usr/bin/env python
"""
Build coco_caption_clip.index & coco_caption_texts.npy from the
Hugging Face 'mscoco' caption dataset.
"""
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import tqdm
import os
import pathlib

# Configuration
OUT_DIR = pathlib.Path(__file__).resolve().parent   # scripts/
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "clip-ViT-B-32"
DEVICE = "cpu"          # Use "mps" on Apple Silicon, "cuda" if you have a GPU
BATCH = 1024           # Adjust for RAM

def main():
    print("Loading COCO captions via Hugging Face datasets...")
    ds = load_dataset("mscoco", "2014")                # train+val 2014 split
    texts = [rec["caption"] for rec in ds["train"]] + \
            [rec["caption"] for rec in ds["validation"]]
    print(f"Loaded {len(texts):,} captions.")

    print(f"Embedding with {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    vecs = model.encode(
        texts,
        batch_size=BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    print("Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    faiss_path = OUT_DIR / "coco_caption_clip.index"
    capt_path = OUT_DIR / "coco_caption_texts.npy"

    print("Saving files...")
    faiss.write_index(index, str(faiss_path))
    np.save(capt_path, np.array(texts, dtype=object))
    
    print("Done!")
    print(f"{faiss_path.name}  → {faiss_path.stat().st_size/1e9:.2f} GB")
    print(f"{capt_path.name}   → {capt_path.stat().st_size/1e9:.2f} GB")

if __name__ == "__main__":
    main()
```

### Run the Script
```bash
python scripts/build_index_from_hf.py
```

## Method B: Manual ZIP Download

### Step 1: Download COCO 2017 Annotations
```bash
curl -L -o annotations_trainval2017.zip \
     http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Step 2: Create Build Script
Create `scripts/build_index_from_json.py`:

```python
#!/usr/bin/env python
"""
Convert raw COCO JSON caption files into FAISS + .npy.
Usage:
    python build_index_from_json.py captions_train2017.json captions_val2017.json
"""
import json
import sys
import pathlib
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "clip-ViT-B-32"
DEVICE = "cpu"     # Change to "mps" or "cuda" as desired
BATCH = 1024

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: build_index_from_json.py <json> [<json> ...]")

    print("Collecting captions from JSON files...")
    captions = []
    for fp in sys.argv[1:]:
        print(f"Processing {fp}...")
        with open(fp, 'r') as f:
            data = json.load(f)
        captions.extend([ann["caption"] for ann in data["annotations"]])
    
    print(f"Total captions: {len(captions):,}")

    print(f"Embedding with {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    vecs = model.encode(
        captions,
        batch_size=BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    out_dir = pathlib.Path(__file__).resolve().parent
    faiss_path = out_dir / "coco_caption_clip.index"
    capt_path = out_dir / "coco_caption_texts.npy"
    
    print("Saving files...")
    faiss.write_index(index, str(faiss_path))
    np.save(capt_path, np.array(captions, dtype=object))

    print("Done!")
    print(f"Files saved to: {out_dir}")
    print(f"{faiss_path.name}  → {faiss_path.stat().st_size/1e9:.2f} GB")
    print(f"{capt_path.name}   → {capt_path.stat().st_size/1e9:.2f} GB")

if __name__ == "__main__":
    main()
```

### Step 3: Run the Script
With these files
![Screenshot of the Gradio demo UI](data/coco.png)
we build the indices
```bash
python scripts/build_index_from_json.py \
       annotations/captions_train2017.json \
       annotations/captions_val2017.json
```
which looks like this
![Screenshot of the Gradio demo UI](data/coco2.png)

## Testing the Demo

After building the index files, test the demo:

```bash
conda activate capstone-gradio-py310
python gradio_demo.py
```

You should see:
```
FAISS index loaded: 591753 vectors, dimension 512
Launching Gradio demo...
```

Open http://127.0.0.1:7860 and you're ready to go!

## Troubleshooting

- **Memory issues**: Reduce the `BATCH` size in the configuration
- **GPU not detected**: Change `DEVICE` to `"cpu"` if you encounter CUDA errors
- **Missing dependencies**: Install required packages with `pip install sentence-transformers faiss-cpu numpy tqdm`
- **File not found**: Ensure the `scripts/` directory exists and you're running from the correct location
