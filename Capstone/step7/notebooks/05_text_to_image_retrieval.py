"""
text_to_image_demo.py
---------------------
• Loads one experiment run (embeddings + metadata)
• Prints the top-K nearest images (domain, score, id) for three example prompts
• Computes text→image Recall@1/5/10 with a low-RAM, chunked routine

Args (optional)
---------------
--root PATH   : path to experiments folder
--run  NAME   : run folder (default = newest with embeddings)
--k    N      : top-K neighbours to show (default 5)

Example
-------
python text_to_image_demo.py
python text_to_image_demo.py --run RN50_20250623_214602
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import open_clip, torch
import pandas as pd
from numpy.linalg import norm

# command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--root",
                    default=r"C:/Users/steph/OneDrive/Desktop/experiments",
                    help="Experiments directory")
parser.add_argument("--run", default=None,
                    help="Run folder name (default = newest run with embeddings)")
parser.add_argument("--k", type=int, default=5,
                    help="Top-K images to display")
args = parser.parse_args()

EXP_ROOT = Path(args.root)
if not EXP_ROOT.exists():
    sys.exit(f"Experiments dir not found: {EXP_ROOT}")

def has_embeddings(p: Path) -> bool:
    return (p / "img_embs.npy").exists() and (p / "txt_embs.npy").exists()

if args.run:
    RUN_DIR = EXP_ROOT / args.run
    if not has_embeddings(RUN_DIR):
        sys.exit(f"{RUN_DIR} lacks img_embs.npy / txt_embs.npy")
else:
    runs = sorted((p for p in EXP_ROOT.iterdir() if has_embeddings(p)),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"No runs with embeddings in {EXP_ROOT}")
    RUN_DIR = runs[-1]

print("Using run:", RUN_DIR.name)

# load embeddings and metadata

IMG_EMB = np.load(RUN_DIR / "img_embs.npy").astype("float32")
IDS     = json.loads((RUN_DIR / "ids.json").read_text())

PARQUET = Path(r"C:/Users/steph/OneDrive/Desktop/data/metadata.parquet")
META    = pd.read_parquet(PARQUET).set_index("id").loc[IDS]

img_norm = IMG_EMB / norm(IMG_EMB, axis=1, keepdims=True)
print(f"Loaded {IMG_EMB.shape[0]:,} image embeddings")

# cache text encoder once
MODEL_NAME   = RUN_DIR.name.split("_")[0]            # "RN50" or "ViT-B-32"
PRETRAIN_TAG = "openai" if MODEL_NAME == "RN50" else "laion2b_s34b_b79k"

model, _, _ = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAIN_TAG, device="cpu"
)
model.eval()
tokenize = open_clip.tokenize

def embed_text(text: str) -> np.ndarray:
    with torch.no_grad():
        tok = tokenize([text])
        vec = model.encode_text(tok).cpu().numpy().astype("float32")
        return vec / norm(vec, axis=1, keepdims=True)

def topk_images(query: str, k: int):
    q_vec = embed_text(query)[0]
    sim   = q_vec @ img_norm.T
    idx   = np.argpartition(-sim, k)[:k]
    idx   = idx[np.argsort(-sim[idx])]
    return idx, sim[idx]

def print_results(query: str, k: int):
    idx, scores = topk_images(query, k)
    print(f"\nQuery: \"{query}\"")
    for rank, (i, s) in enumerate(zip(idx, scores), 1):
        row = META.iloc[i]
        print(f"{rank:2d}.  domain={row.domain:6}  score={s:.3f}  id={row.name}")

# sample prompts
SAMPLES = [
    "a red sports car on a city street",
    "two dogs playing in the snow",
    "cyberpunk city at night with neon lights",
]

for prompt in SAMPLES:
    print_results(prompt, args.k)

# Recall@K evaluation (chunked, low-RAM)
txt_norm = np.load(RUN_DIR / "txt_embs.npy").astype("float32")
txt_norm = txt_norm / norm(txt_norm, axis=1, keepdims=True)

def text_to_image_recall(k: int = 1, chunk: int = 2_000) -> float:
    n = txt_norm.shape[0]
    hits = np.zeros(n, dtype=bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = txt_norm[s:e] @ img_norm.T
        topk = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(s, e)[:, None]
        hits[s:e] = np.any(topk == rows, axis=1)
    return hits.mean() * 100.0

print("\nRecall@K on this run:")
for k in (1, 5, 10):
    print(f"  Recall@{k}: {text_to_image_recall(k):5.2f} %")

print("\nDone.")


"""
The output will show the top-K images for each sample prompt, along with their domain, similarity score, and ID.
It will also compute and display the Recall@1, Recall@5, and Recall@10 for the text-to-image retrieval task.

Here is an example output:
Using run: RN50_20250623_214602
Loaded 2,000 image embeddings
C:\Users\steph\AppData\Local\Programs\Python\Python313\Lib\site-packages\open_clip\factory.py:388: UserWarning: These pretrained weights were trained with QuickGELU activation but the model config does not have that enabled. Consider using a model config with a "-quickgelu" suffix or enable with a flag.
  warnings.warn(

Query: "a red sports car on a city street"
 1.  domain=sd      score=0.245  id=sd_53939
 2.  domain=coco    score=0.214  id=co_425248
 3.  domain=coco    score=0.208  id=co_651152
 4.  domain=coco    score=0.208  id=co_757318
 5.  domain=coco    score=0.206  id=co_303415

Query: "two dogs playing in the snow"
 1.  domain=coco    score=0.250  id=co_128300
 2.  domain=flickr  score=0.232  id=fl_64289
 3.  domain=coco    score=0.229  id=co_549738
 4.  domain=coco    score=0.227  id=co_496420
 5.  domain=flickr  score=0.208  id=fl_67871

Query: "cyberpunk city at night with neon lights"
 1.  domain=coco    score=0.193  id=co_769416
 2.  domain=coco    score=0.193  id=co_705543
 3.  domain=sd      score=0.191  id=sd_13997
 4.  domain=coco    score=0.185  id=co_452509
 5.  domain=sd      score=0.172  id=sd_76838

Recall@K on this run:
  Recall@1: 48.15 %
  Recall@5: 76.50 %
  Recall@10: 85.20 %

Done.

"""