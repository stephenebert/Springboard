# encoder_clip.py
"""
Safe, single‑process CLIP text embedder for macOS
------------------------------------------------
* CPU‑only (avoids Metal / GPU issues)
* Forces Torch / BLAS to single thread – fixes semaphore seg‑faults
* Caches tokenizer & model with @lru_cache (loads once)
"""

# ---------- single‑thread guards (MUST be first!) -----------------
import os
os.environ["OMP_NUM_THREADS"] = "1"     # NumPy / OpenBLAS / Accelerate
os.environ["MKL_NUM_THREADS"] = "1"     # MKL, if linked

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# ------------------------------------------------------------------

from functools import lru_cache
from typing import List
import numpy as np
from transformers import CLIPTokenizer, CLIPModel

DEVICE = "cpu"       # keep "cpu" on macOS
BATCH  = 16          # embed this many captions at once


@lru_cache(maxsize=1)
def _clip_components():
    """Load tokenizer & model once, cache them."""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings with CLIP and return an array of shape (N, 512).

    * Runs single‑process, single‑thread – safe on macOS
    * Vectors are L2‑normalized (unit length)
    """
    tokenizer, model = _clip_components()
    out = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            toks  = tokenizer(batch,
                              padding=True,
                              truncation=True,
                              return_tensors="pt").to(DEVICE)
            feats = model.get_text_features(**toks)          # (B, 512)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.cpu().numpy())

    return np.vstack(out)
