#!/usr/bin/env python3
"""
Step 8 Index Builder
Builds an IVF-Flat FAISS index from precomputed embeddings (image & text) and
outputs both the index file and a JSONL metadata file for retrieval.

Usage example (run inside Docker container):

    python scripts/build_index_step8.py \
        --h5 /data/embeddings_full.h5 \
        --meta /data/metadata.parquet \
        --split train \
        --out_dir /data/faiss-indexes \
        --nlist 1024

"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import h5py
import faiss

def load_embeddings(h5_path: str, mask: np.ndarray, img_key: str, txt_key: str):
    """
    Load and slice embeddings from an HDF5 container by boolean mask.
    """
    with h5py.File(h5_path, 'r') as f:
        # gather masked indices
        idxs = np.nonzero(mask)[0]
        # read datasets
        img_ds = f[img_key]
        txt_ds = f[txt_key]
        # slice out only the rows in this split
        img_embs = img_ds[idxs]
        txt_embs = txt_ds[idxs]
    return img_embs, txt_embs, idxs

def main(h5_path: str, meta_path: str, split: str, out_dir: str, nlist: int,
         img_key: str, txt_key: str):
    # load metadata
    meta_df = pd.read_parquet(meta_path)
    if split not in meta_df['split'].unique():
        raise ValueError(f"Split '{split}' not found in metadata {meta_path}")
    # boolean mask of rows in desired split
    mask = meta_df['split'] == split
    num = mask.sum()
    print(f"[Step 8] Loading metadata...\n---> {num} rows in split='{split}'")

    # load embeddings
    print(f"[Step 8] Loading embeddings (this may take a minute)...")
    img_embs, txt_embs, idxs = load_embeddings(h5_path, mask.values, img_key, txt_key)

    # prepare output dir
    os.makedirs(out_dir, exist_ok=True)
    dim = img_embs.shape[1]

    # train IVF-Flat index on image embeddings (inner-product)
    print(f"[Step 8] Training IVF index (nlist = {nlist} )...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(img_embs)
    # add embeddings with original global IDs for lookup
    index.add_with_ids(img_embs, idxs.astype(np.int64))

    # write index
    idx_fname = os.path.join(out_dir, f"ivf_flat_{nlist}.index")
    faiss.write_index(index, idx_fname)
    print(f"Wrote index --> {idx_fname}")

    # dump metadata JSONL (image_path + caption) aligned to global IDs
    recs = []
    sub_df = meta_df.loc[mask, ['image_path', 'caption']]
    sub_df = sub_df.reset_index(drop=False).rename(columns={'index': 'id'})
    # reorganize into list of dicts
    for _, row in sub_df.iterrows():
        recs.append({'id': int(row['id']),
                     'image_path': row['image_path'],
                     'caption': row['caption']})
    meta_fname = os.path.join(out_dir, f"metadata_{split}.jsonl")
    with open(meta_fname, 'w') as outf:
        for rec in recs:
            json.dump(rec, outf)
            outf.write("\n")
    print(f"Wrote metadata --> {meta_fname}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build Step8 FAISS IVF-Flat index + metadata JSONL.")
    parser.add_argument('--h5', required=True,
                        help='Path to HDF5 with image/text embeddings')
    parser.add_argument('--meta', required=True,
                        help='Path to metadata Parquet file (must have image_path, caption, split columns)')
    parser.add_argument('--split', default='train',
                        help='Which split to index (e.g. train)')
    parser.add_argument('--out_dir', required=True,
                        help='Directory to write index and metadata files')
    parser.add_argument('--nlist', type=int, default=1024,
                        help='Number of Voronoi cells (coarse clusters) for IVF')
    parser.add_argument('--img_key', default='image_embeddings',
                        help='HDF5 dataset key for image embeddings')
    parser.add_argument('--txt_key', default='text_embeddings',
                        help='HDF5 dataset key for text embeddings')
    args = parser.parse_args()
    main(args.h5, args.meta, args.split, args.out_dir, args.nlist,
         args.img_key, args.txt_key)
