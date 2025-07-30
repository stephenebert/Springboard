#!/usr/bin/env python
# evaluate_baseline.py
#
# Evaluate a retrieval pipeline on COCO-captions-val2017.
# Supports SBERT / MM-Embed / ImageBind encoders and FAISS engines
# (ivf_flat | ivfpq | hnsw).

import argparse, json, time, pathlib
import numpy as np
import faiss
from tqdm import tqdm

# ---------- encoders -------------------------------------------------
from encoder_sbert      import embed_texts as embed_sbert
from encoder_mmembed    import embed_texts as embed_mm
from encoder_imagebind  import embed_texts as embed_ib  # NEW

ENCODER_REGISTRY = {
    "sbert"     : embed_sbert,
    "mmembed"   : embed_mm,
    "imagebind" : embed_ib,
}

# ---------- helpers --------------------------------------------------
def load_captions(coco_json, limit):
    caps = []
    with open(coco_json, "r") as fh:
        data = json.load(fh)
    for ann in data["annotations"][:limit]:
        caps.append(ann["caption"].strip())
    return caps                        # len = limit

def load_index(path, ef_search=None):
    idx = faiss.read_index(path)
    if (idx.__class__.__name__.startswith("IndexHNSW")) and ef_search:
        faiss.ParameterSpace().set_index_parameters(idx, f"efSearch={ef_search}")
    return idx

def compute_recall_at_k(I_all, top_k):
    """
    Ground-truth: caption i ↔ image i
    I_all is stacked (N, top_k) index matrix.
    """
    N = I_all.shape[0]
    correct = (I_all[:, :top_k] == np.arange(N)[:, None]).any(axis=1).sum()
    return correct / N

# ---------- main -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",  required=True, help="captions_val2017.json")
    ap.add_argument("--index", required=True, help="pre-built FAISS index")
    ap.add_argument("--encoder", choices=["sbert","mmembed","imagebind"],
                    default="sbert")
    ap.add_argument("--engine",  choices=["ivf_flat","ivfpq","hnsw"],
                    default="hnsw")
    ap.add_argument("--limit",   type=int, default=5000)
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--top_k",   type=int, default=1)
    ap.add_argument("--ef_search", type=int, default=64)
    args = ap.parse_args()

    embed_fn   = ENCODER_REGISTRY[args.encoder]
    captions   = load_captions(args.json, args.limit)
    print(f"▶ Loaded {len(captions):,} captions (args.encoder='{args.encoder}')")

    # -- embedding ----------------------------------------------------
    embed_t0   = time.time()
    preds, embed_times = [], []
    idx        = load_index(args.index, ef_search=args.ef_search)
    for i in tqdm(range(0, len(captions), args.batch), desc="embedding"):
        batch  = captions[i:i+args.batch]
        t0     = time.time()
        vec    = embed_fn(batch)                 # torch.Tensor | np.ndarray
        embed_times.append(time.time()-t0)

        if hasattr(vec, "detach"):               # torch → numpy
            vec = vec.cpu().numpy()
        vec = vec.astype("float32")
        D,I   = idx.search(vec, args.top_k)
        preds.append((D,I))                      # keep tuple  (for SBERT/MM)
    embed_elapsed = sum(embed_times)

    # -------- concat search results ----------------------------------
    if isinstance(preds[0], tuple):              # tuple style (old)
        D_all = np.concatenate([d for d,_ in preds])
        I_all = np.concatenate([i for _,i in preds])
    else:                                        # flat list style (ImageBind)
        D_all = np.concatenate(preds[0::2])      # even  → distances
        I_all = np.concatenate(preds[1::2])      # odd   → ids

    # -------- metrics ------------------------------------------------
    recall = compute_recall_at_k(I_all, args.top_k)
    latency_ms = (time.time()-embed_t0)*1000/len(captions)
    embed_ms   = embed_elapsed*1000/len(captions)

    print(f"\nRecall@{args.top_k}: {recall:0.4f}   |  engine={args.engine}")
    print(f"Avg latency: {latency_ms:5.2f} ms/query  (embed: {embed_ms:4.1f} ms)")

if __name__ == "__main__":
    main()
