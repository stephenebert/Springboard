#!/usr/bin/env python3
"""
Embed COCO captions with any encoder_* module.

Examples
--------
python embed_coco.py \
  --json   /path/to/captions_val2017.json \
  --out    data/embeds_mmembed.npy \
  --limit  5000 \
  --batch  128 \
  --encoder encoder_mmembed \
  --verbose
"""
import argparse, importlib, json, time, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
def load_coco_caps(path: str, limit: int | None = None) -> list[str]:
    with open(path, "r") as f:
        anns = json.load(f)["annotations"]
    return [ann["caption"] for ann in anns][:limit]

# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",   required=True, help="COCO captions JSON")
    parser.add_argument("--out",    required=True, help="output .npy filename")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--encoder",default="encoder_clip",
                        help="module providing embed_texts() (default: encoder_clip)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # dynamic import ---------------------------------------------------------
    try:
        enc_mod     = importlib.import_module(args.encoder)
        embed_texts = enc_mod.embed_texts
    except Exception as e:
        sys.exit(f"❌  could not import {args.encoder}: {e}")

    caps  = load_coco_caps(args.json, args.limit)
    if args.verbose:
        print(f"► Loaded {len(caps):,} captions — embedding…")

    t0, batches, embs = time.time(), [], []
    for i in tqdm(range(0, len(caps), args.batch)):
        sub = caps[i : i + args.batch]
        embs.append(embed_texts(sub))
    embs = np.vstack(embs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, embs)
    if args.verbose:
        dt = time.time() - t0
        rate = len(caps) / dt
        print(f"✓ Saved {embs.shape} → {args.out}   ({dt:0.1f}s, {rate:,.0f} cap/s)")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
