#!/usr/bin/env python3
# encoder_mmembed.py

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ——— module‐wide defaults ———————————————————————————————————————————————
_MODEL_DIR  = Path(__file__).parent / "models" / "mme-lite"
_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_BATCH_SIZE = 256

# load once
_model = SentenceTransformer(str(_MODEL_DIR), device=_DEVICE)
_model.eval()

@torch.inference_mode()
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Used by evaluate_baseline.py: embeds a batch of captions → (N, D) float32
    """
    emb = _model.encode(
        texts,
        batch_size   = _BATCH_SIZE,
        convert_to_tensor   = True,
        normalize_embeddings = True,   # cosine-sim = dot-prod
        device       = _DEVICE,
        show_progress_bar = False,
    )
    return emb.cpu().numpy().astype("float32")


def _load_captions(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text())
    return [ann["caption"].strip() for ann in data["annotations"]]


def main():
    global _DEVICE, _BATCH_SIZE, _model

    p = argparse.ArgumentParser(
        description="Embed COCO captions with MME-Lite and dump to a .npy file"
    )
    p.add_argument("--json",   required=True,
                   help="path to coco/annotations/captions_val2017.json")
    p.add_argument("--out",    required=True,
                   help="where to write the .npy embeddings")
    p.add_argument("--batch",  type=int, default=_BATCH_SIZE,
                   help="batch size for encoding")
    p.add_argument("--device", choices=["cpu","cuda"], default=None,
                   help="force device (otherwise auto-detect)")
    p.add_argument("--debug",  action="store_true",
                   help="print per-batch debug info")
    args = p.parse_args()

    # override module‐level DEVICE & BATCH_SIZE if requested
    if args.device:
        _DEVICE = args.device
        _model  = SentenceTransformer(str(_MODEL_DIR), device=_DEVICE)
        _model.eval()
    _BATCH_SIZE = args.batch

    captions = _load_captions(Path(args.json))
    if args.debug:
        print(f"▶ Embedding {len(captions)} captions → device={_DEVICE}, batch={_BATCH_SIZE}")

    t0     = time.time()
    chunks = []
    for i in range(0, len(captions), _BATCH_SIZE):
        batch = captions[i : i + _BATCH_SIZE]
        emb   = embed_texts(batch)
        chunks.append(emb)
        if args.debug:
            print(f"  • batch {i:5d}-{i+len(batch):5d} → {emb.shape}")

    all_emb = np.vstack(chunks)
    np.save(args.out, all_emb)
    print(f"✓ wrote {args.out}   shape={all_emb.shape}   time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
