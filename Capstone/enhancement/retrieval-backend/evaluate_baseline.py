#!/usr/bin/env python
# evaluate_baseline.py
#
# Single-stage retrieval evaluation on COCO-captions-val2017.
# Encoders:  SBERT | CLIP | MM-Embed | ImageBind
# FAISS engines: ivf_flat | ivfpq | hnsw

from __future__ import annotations
import argparse, json, pathlib, time
from typing import List, Tuple

import numpy as np
import faiss
from tqdm import tqdm

# ─────────────────────────────── helpers ──────────────────────────────────────
def load_captions(coco_json: str | pathlib.Path, limit: int) -> List[str]:
    with open(coco_json, "r") as fh:
        data = json.load(fh)
    return [ann["caption"].strip() for ann in data["annotations"][:limit]]

def load_index(path: str | pathlib.Path, ef_search: int | None) -> faiss.Index:
    idx = faiss.read_index(str(path))
    if idx.__class__.__name__.startswith("IndexHNSW") and ef_search:
        faiss.ParameterSpace().set_index_parameters(idx,
                                                    f"efSearch={ef_search}")
    return idx

def compute_recall_at_k(I: np.ndarray, k: int) -> float:
    """Ground truth for caption i is id=i."""
    gt = np.arange(I.shape[0])[:, None]
    return float((I[:, :k] == gt).any(axis=1).mean())

# ──────────────────────── lazy-import encoders ────────────────────────────────
def get_embed_fn(name: str):
    if name == "sbert":
        from encoder_sbert import embed_texts as fn
    elif name == "clip":
        from encoder_clip import embed_texts as fn
    elif name == "mmembed":
        from encoder_mmembed import embed_texts as fn
    elif name == "imagebind":
        from encoder_imagebind import embed_texts as fn
    else:
        raise ValueError(f"Unknown encoder {name}")
    return fn

# ─────────────────────────────── main ─────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",   required=True,
                    help="captions_val2017.json")
    ap.add_argument("--index",  required=True,
                    help="pre-built FAISS index")
    ap.add_argument("--encoder", choices=["sbert","clip","mmembed","imagebind"],
                    default="sbert")
    ap.add_argument("--engine",  choices=["ivf_flat","ivfpq","hnsw"],
                    default="hnsw")
    ap.add_argument("--limit",   type=int, default=5_000)
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--top_k",   type=int, default=1)
    ap.add_argument("--ef_search", type=int, default=64)
    ap.add_argument("--out_tsv", default=None,
                    help="write recall/latency to this TSV")
    args = ap.parse_args()

    captions = load_captions(args.json, args.limit)
    print(f"▶ Loaded {len(captions):,} captions (encoder='{args.encoder}')")

    embed_fn = get_embed_fn(args.encoder)
    index    = load_index(args.index, args.ef_search)

    # ── embed + search ────────────────────────────────────────────────────────
    t0 = time.time()
    embed_ms: list[float] = []
    preds: list[Tuple[np.ndarray, np.ndarray]] = []

    for i in tqdm(range(0, len(captions), args.batch), desc="embedding"):
        batch   = captions[i:i+args.batch]
        t_start = time.time()
        vec     = embed_fn(batch)

        if hasattr(vec, "detach"):               # torch.Tensor → numpy
            vec = vec.cpu().numpy()
        vec = vec.astype("float32")

        D, I = index.search(vec, args.top_k)
        preds.append((D, I))                     # tuple for all encoders
        embed_ms.append((time.time()-t_start)*1000)

    embed_mean = float(np.mean(embed_ms))        # per-query ms
    latency_ms = (time.time()-t0)*1000/len(captions)

    # ── stack results ─────────────────────────────────────────────────────────
    D_all = np.concatenate([d for d,_ in preds], axis=0)
    I_all = np.concatenate([i for _,i in preds], axis=0)

    recall = compute_recall_at_k(I_all, args.top_k)

    print(f"\nRecall@{args.top_k}: {recall:0.4f}   |  engine={args.engine}")
    print(f"Avg latency: {latency_ms:5.2f} ms/query  (embed: {embed_mean:4.1f} ms)")

    if args.out_tsv:
        pathlib.Path(args.out_tsv).write_text(
            f"{args.encoder}\t{args.engine}\t{args.limit}\t{args.top_k}\t"
            f"{recall:.4f}\t{latency_ms:.2f}\t{embed_mean:.2f}\n"
        )
        print(f"✔ Results appended → {args.out_tsv}")

if __name__ == "__main__":
    main()
