# encoder_mmembed.py

from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

_MODEL_DIR = Path(__file__).parent / "models" / "mme-lite"
_device     = "cuda" if torch.cuda.is_available() else "cpu"
_model      = SentenceTransformer(str(_MODEL_DIR), device=_device)
_model.eval()                

@torch.inference_mode()
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Args:
        texts (list[str])  – batch of sentences / captions
    Returns:
        (N, D) float32 numpy array  (D ≈ 1024 for mxbai‑embed‑large‑v1)
    """
    emb = _model.encode(
        texts,
        batch_size = 64,
        convert_to_tensor = True,
        normalize_embeddings = True,   # cosine ≡ dot‑prod
        device = _device,
        show_progress_bar = False,
    )
    return emb.cpu().numpy().astype("float32")
