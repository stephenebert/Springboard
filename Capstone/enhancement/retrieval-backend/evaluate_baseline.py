#!/usr/bin/env python3
import argparse
import json
import time

import faiss
import numpy as np

def load_coco_caps(path, limit=None):
    with open(path, "r") as f:
        data = json.load(f)
    caps = [ann["caption"] for ann in data["annotations"]]
    return caps if limit is None else caps[:limit]

def main():
    p = argparse.ArgumentParser(
        description="Evaluate FAISS baseline (IVF‑Flat / IVF‑PQ / HNSW) on COCO captions"
    )
    p.add_argument("--json",      required=True, help="COCO captions JSON file")
    p.add_argument("--limit",     type=int,    default=1000,
                   help="How many captions to load")
    p.add_argument("--engine",    choices=["ivf_flat","ivfpq","hnsw"],
                   required=True, help="FAISS index type")
    p.add_argument("--index",     help="Path to .faiss index (for ivfpq or hnsw)")
    p.add_argument("--nlist",     type=int,    default=512,
                   help="Number of buckets (IVF‑PQ only)")
    p.add_argument("--pq_m",      type=int,    default=32,
                   help="Number of sub‑quantizers (IVF‑PQ only)")
    p.add_argument("--ef_search", type=int,    default=32,
                   help="HNSW efSearch (accuracy/speed trade‑off)")
    p.add_argument("--batch",     type=int,    default=32,
                   help="Batch size for embedding")
    p.add_argument("--top_k",     type=int,    default=10,
                   help="Recall@K")
    p.add_argument("--workers",   type=int,    default=0,
                   help="Number of workers for embedding (future use)")
    p.add_argument("--encoder",   choices=["sbert","mmembed"],
                   default="sbert", help="Which text encoder to use")
    args = p.parse_args()

    # import the right embed_texts function
    if args.encoder == "sbert":
        from encoder_sbert import embed_texts
    else:
        from encoder_mmembed import embed_texts

    # 1) Load & embed COCO captions
    caps = load_coco_caps(args.json, limit=args.limit)
    print(f"Loaded {len(caps)} captions from {args.json}")

    all_embs = []
    t0 = time.time()
    for i in range(0, len(caps), args.batch):
        batch = caps[i : i + args.batch]
        embs = embed_texts(batch)               # returns np.ndarray, shape=(len(batch), dim)
        all_embs.append(embs)
    X = np.vstack(all_embs)
    print(f"Embedding done in {time.time()-t0:.2f}s  (dim={X.shape[1]})")

    # 2) Build / load FAISS index
    if args.engine == "ivf_flat":
        d = X.shape[1]
        index = faiss.index_factory(d, "IVF20,Flat")
        print("Training IVF‑Flat index...")
        index.train(X)
        index.add(X)

    elif args.engine == "ivfpq":
        d = X.shape[1]
        spec = f"IVF{args.nlist},PQ{args.pq_m}"
        index = faiss.index_factory(d, spec)
        print(f"Training IVF‑PQ index ({spec})…")
        index.train(X)
        index.add(X)

    else:  # hnsw
        assert args.index, "Must pass --index for HNSW"
        index = faiss.read_index(args.index)
        index.hnsw.efSearch = args.ef_search
        print(f"Loaded HNSW index from {args.index} (efSearch={args.ef_search})")

    # 3) Run queries & measure Recall@K
    print("Running queries…")
    start = time.time()
    correct = 0
    for i, cap in enumerate(caps):
        q_emb = embed_texts([cap])  # shape (1, dim)
        _, I = index.search(q_emb, args.top_k)
        if i in I[0]:
            correct += 1

    recall = correct / len(caps)
    latency_ms = (time.time() - start) / len(caps) * 1000

    print(f"Recall@{args.top_k}: {recall:.4f}   |  engine={args.engine}")
    print(f"Avg latency: {latency_ms:.2f} ms/query")


if __name__ == "__main__":
    main()
