#!/usr/bin/env python3
# generate_coco_embeddings.py
# Place this in benchmark/, then run: python generate_coco_embeddings.py

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
import os

# Files in this folder
TEXTS_FILE  = "coco_caption_texts.npy"
OUT_EMBEDS  = "coco_caption_clip.npy"
BATCH_SIZE  = 512

# Device (M4‑Max metal / CPU)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[*] Using device: {DEVICE}")

def main():
    assert os.path.isfile(TEXTS_FILE), f"{TEXTS_FILE} not found!"
    texts = np.load(TEXTS_FILE, allow_pickle=True)
    n = len(texts)
    print(f"[*] Loaded {n:,} captions.")

    # Load CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model     = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    model.eval()

    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH_SIZE), desc="Embedding"):
            batch_texts = list(texts[i : i + BATCH_SIZE])
            tokens = tokenizer(batch_texts, return_tensors="pt",
                               padding=True, truncation=True).to(DEVICE)
            out = model(**tokens)
            # pooled_output is [batch_size, hidden_dim]
            embeds = out.pooler_output.cpu().numpy()
            all_embeds.append(embeds)

    all_embeds = np.vstack(all_embeds)
    print(f"[+] Final embeddings shape: {all_embeds.shape}")
    np.save(OUT_EMBEDS, all_embeds)
    print(f"Saved embeddings to {OUT_EMBEDS}")

if __name__ == "__main__":
    main()
