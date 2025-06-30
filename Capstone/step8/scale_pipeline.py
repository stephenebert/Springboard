# Scale up embeddings to the full dataset in float16, writing to HDF5
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import open_clip
from open_clip import tokenize
from tqdm import tqdm
import h5py

PARQUET     = Path(r"C:\Users\steph\OneDrive\Desktop\data\metadata.parquet")
OUT_FILE    = Path(r"C:\Users\steph\OneDrive\Desktop\step7\experiments\full\embeddings_full.h5")
BACKBONE    = "ViT-B-32"
PRETRAINED  = "laion2b_s34b_b79k"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
CHUNK_SIZE  = 2000
NUM_WORKERS = 4

class ChunkDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert("RGB")
        img = self.transform(img)
        # return the raw caption string, not tokens
        return img, row.caption

def main():
    # load metadata
    print(f"Loading metadata from {PARQUET}")
    meta = pd.read_parquet(PARQUET)
    meta = meta[meta.split == "train"].reset_index(drop=True)
    N = len(meta)
    print(f"Found {N:,} train rows")

    # ensure output directory exists
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings to {OUT_FILE}")

    # load model & transforms
    print(f"Loading model {BACKBONE} on {DEVICE}")
    model, _, _ = open_clip.create_model_and_transforms(
        BACKBONE, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    if DEVICE == "cuda":
        model = model.half()

    transform = T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466,0.4578275,0.40821073),
                    (0.26862954,0.26130258,0.27577711)),
    ])

    # create HDF5 datasets (float16)
    img_dim = model.visual.output_dim
    txt_dim = model.text_projection.shape[1]

    print(f"Creating HDF5 file {OUT_FILE}")
    hf = h5py.File(OUT_FILE, "w")
    img_ds = hf.create_dataset(
        "image_embeddings",
        shape=(N, img_dim),
        dtype="float16",
        chunks=(CHUNK_SIZE, img_dim),
        compression="lzf"
    )
    txt_ds = hf.create_dataset(
        "text_embeddings",
        shape=(N, txt_dim),
        dtype="float16",
        chunks=(CHUNK_SIZE, txt_dim),
        compression="lzf"
    )

    # 5) chunked embedding loop
    print("Starting chunked embedding...")
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        sub_df = meta.iloc[start:end].reset_index(drop=True)

        loader = DataLoader(
            ChunkDataset(sub_df, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE=="cuda")
        )

        img_buf, txt_buf = [], []
        with torch.no_grad():
            for images, captions in tqdm(loader, desc=f"Chunk {start}-{end}"):
                images = images.to(DEVICE)
                # **batch‐tokenize** the list of raw captions
                text_tokens = tokenize(captions).to(DEVICE)

                if DEVICE == "cuda":
                    images = images.half()
                    text_tokens = text_tokens.half()

                ie = model.encode_image(images)
                ie = ie / ie.norm(dim=-1, keepdim=True)

                te = model.encode_text(text_tokens)
                te = te / te.norm(dim=-1, keepdim=True)

                img_buf.append(ie.cpu().half().numpy())
                txt_buf.append(te.cpu().half().numpy())

        img_ds[start:end, :] = np.vstack(img_buf)
        txt_ds[start:end, :] = np.vstack(txt_buf)

    hf.flush()
    hf.close()
    print("All embeddings written to HDF5 (float16)")

if __name__ == "__main__":
    main()
