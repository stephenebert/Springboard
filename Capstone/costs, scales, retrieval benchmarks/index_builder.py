#!/usr/bin/env python3
"""
Build a FAISS index from a NumPy `.npy` embeddings file.

Usage
-----
python index_builder.py \
    --embeds data/embeds_val2017.npy \
    --out    hnsw_val2017.faiss \
    --engine hnsw           \      # or ivf_flat / ivfpq
    --ef_search 64          \      # HNSW only (optional)
    --nlist 256             \      # IVF‑PQ / IVF‑Flat only  (optional)
    --pq_m 64                     # IVF‑PQ only (optional)
"""
# index_builder.py 
import argparse, pathlib, faiss, numpy as np, tqdm

def load_embeds(path: str) -> np.ndarray:
    xb = np.load(path).astype("float32")
    print(f"Loaded {len(xb):,} vectors  (dim={xb.shape[1]})")
    return xb

def build_ivfpq(dim: int, nlist: int, pq_m: int) -> faiss.IndexIVFPQ:
    quantizer = faiss.IndexFlatL2(dim)          # coarse quantizer
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8)  # 8‑bit sub‑quant
    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeds", required=True, help="*.npy file from embed_coco.py")
    parser.add_argument("--out",    required=True, help="output *.faiss file")
    parser.add_argument("--engine", choices=["hnsw", "ivf_flat", "ivfpq"], required=True)
    parser.add_argument("--nlist",  type=int, default=512,  help="IVF lists (ivfpq)")
    parser.add_argument("--pq_m",   type=int, default=32,   help="PQ sub‑vectors (ivfpq)")
    parser.add_argument("--ef_search", type=int, default=64,help="HNSW efSearch")
    args = parser.parse_args()

    xb   = load_embeds(args.embeds)
    dim  = xb.shape[1]

    # engine dispatch
    if args.engine == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)        # M=32
        index.hnsw.efSearch = args.ef_search

    elif args.engine == "ivf_flat":
        index = faiss.index_factory(dim, f"IVF{args.nlist},Flat")
        print("Training index …"); index.train(xb)

    elif args.engine == "ivfpq":
        index = build_ivfpq(dim, args.nlist, args.pq_m)
        print("Training IVF‑PQ ...")
        # Faiss wants a contiguous array; tqdm for simple progress readout.
        index.train(xb)

    # Add vectors (with mini‑batches so memory stays low)
    print("Adding vectors ...")
    for start in tqdm.tqdm(range(0, len(xb), 10_000)):
        index.add(xb[start:start+10_000])

    faiss.write_index(index, args.out)
    print(f"✓ written {pathlib.Path(args.out).name}")

if __name__ == "__main__":
    main()
