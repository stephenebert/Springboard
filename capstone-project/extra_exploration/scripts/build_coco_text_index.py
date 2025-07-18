#!/usr/bin/env python
"""
Build a FAISS index over all COCO captions (train2017).
Outputs:
  - coco_caption_clip.index        (FAISS index)
  - coco_caption_texts.npy         (NumPy array of captions, aligned with index)
"""

import json, os, numpy as np, faiss, tqdm
from sentence_transformers import SentenceTransformer

ann_path = (
    "/Users/steph/Library/CloudStorage/OneDrive-Personal/Desktop/"
    "Springboard/Springboard/Capstone/step2/data/coco/annotations/"
    "captions_train2017.json"
)
out_index = "coco_caption_clip.index"
out_texts = "coco_caption_texts.npy"

print("▶ Loading COCO captions JSON …")
with open(ann_path) as f:
    data = json.load(f)

captions = [ann["caption"] for ann in data["annotations"]]
print(f"✔ Loaded {len(captions):,} captions")

print("▶ Encoding captions with CLIP (ViT-B/32) …")
clip_model = SentenceTransformer("clip-ViT-B-32")
vectors = clip_model.encode(
    captions,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

print("Building FAISS index ...")
index = faiss.IndexFlatIP(vectors.shape[1])  # cosine b/c embeddings are normalized
index.add(vectors)
faiss.write_index(index, out_index)
np.save(out_texts, np.array(captions))

print(f"Saved index → {out_index}")
print(f"Saved captions → {out_texts}")
