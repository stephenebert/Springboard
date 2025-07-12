#!/usr/bin/env python
"""
Scale-up CLIP embeddings to the full dataset and write them
to an HDF5 container in float16.
"""

from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import open_clip
from open_clip import tokenize

# Dataset helper
class ChunkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filename"]).convert("RGB")
        img = self.transform(img)
        return img, row["caption"]

# Main embedding routine
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metadata", required=True,
                        help="Parquet file with image_path/filename + caption + split")
    parser.add_argument("-i", "--images_root", required=True,
                        help="Root directory containing the images on disk")
    parser.add_argument("-o", "--output", required=True,
                        help="Output HDF5 path (will be created)")
    parser.add_argument("--backbone", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2_000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", default="train",
                        help="Which split column to embed: train/val/test")
    args = parser.parse_args()

    # 1
    print(f"Loading metadata from {args.metadata}")
    meta = pd.read_parquet(args.metadata)

    if "filename" not in meta.columns and "image_path" in meta.columns:
        meta = meta.rename(columns={"image_path": "filename"})


    if args.split:
        meta = meta[meta["split"] == args.split].reset_index(drop=True)
    print(f"Found {len(meta):,} rows in split='{args.split}'")

    # 2)
    # prepend images_root → absolute paths
    meta["filename"] = meta["filename"].apply(
        lambda p: str(Path(args.images_root) / p)
    )

    # 3)
    print(f"Loading {args.backbone} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.backbone, pretrained=args.pretrained, device=args.device
    )
    model.eval()
    if args.device.startswith("cuda"):
        model.half()

    # overwrite mean/std with OpenAI values
    transform = T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])

    img_dim = model.visual.output_dim
    txt_dim = model.text_projection.shape[1]

    # 4
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating HDF5 container at {out_path}")
    h5f = h5py.File(out_path, "w")

    img_ds = h5f.create_dataset(
        "image_embeddings",
        shape=(len(meta), img_dim),
        dtype="float16",
        chunks=(args.chunk_size, img_dim),
        compression="lzf"
    )
    txt_ds = h5f.create_dataset(
        "text_embeddings",
        shape=(len(meta), txt_dim),
        dtype="float16",
        chunks=(args.chunk_size, txt_dim),
        compression="lzf"
    )

    # 5
    for start in range(0, len(meta), args.chunk_size):
        end = min(start + args.chunk_size, len(meta))
        sub_df = meta.iloc[start:end].reset_index(drop=True)

        loader = DataLoader(
            ChunkDataset(sub_df, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )

        img_buf, txt_buf = [], []
        with torch.no_grad():
            for images, captions in tqdm(loader, desc=f"Embedding {start}-{end}"):
                images = images.to(args.device)
                text_tok = tokenize(captions).to(args.device)

                if args.device.startswith("cuda"):
                    images = images.half()

                ie = model.encode_image(images)
                ie /= ie.norm(dim=-1, keepdim=True)

                te = model.encode_text(text_tok)
                te /= te.norm(dim=-1, keepdim=True)

                img_buf.append(ie.cpu().half().numpy())
                txt_buf.append(te.cpu().half().numpy())

        img_ds[start:end] = np.vstack(img_buf)
        txt_ds[start:end] = np.vstack(txt_buf)

    h5f.flush()
    h5f.close()
    print("All embeddings written: done.")


if __name__ == "__main__":
    main()
