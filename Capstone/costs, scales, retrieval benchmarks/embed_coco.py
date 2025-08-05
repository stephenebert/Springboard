#!/usr/bin/env python
# embed_coco.py
#
# Usage example:
#   python embed_coco.py \
#       --json data/coco/annotations/captions_val2017.json \
#       --out  data/embeds_imagebind.npy \
#       --limit 5000 --batch 128 \
#       --encoder imagebind --verbose

import argparse, json, numpy as np, pathlib, time
from tqdm import tqdm

# helpers
def load_coco_caps(path, limit=None):
    """Return a list[str] of captions from COCO‑style JSON."""
    with open(path, "r") as f:
        caps = [ann["caption"] for ann in json.load(f)["annotations"]]
    return caps[:limit] if limit else caps


def save_embeds(path, arr):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype("float32"))
    print(f"✓ wrote {path} shape={arr.shape} dtype=float32")


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",    required=True,
                        help="COCO captions JSON")
    parser.add_argument("--out",     required=True,
                        help="output .npy file for embeddings")
    parser.add_argument("--encoder", choices=["clip", "sbert", "mmembed",
                                              "imagebind"],
                        default="clip")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # pick the encoder
    if args.encoder == "clip":
        from encoder_clip import embed_texts as embed_fn
    elif args.encoder == "sbert":
        from encoder_sbert import embed_texts as embed_fn
    elif args.encoder == "mmembed":
        from encoder_mmembed import embed_texts as embed_fn
    elif args.encoder == "imagebind":
        # ← text‑only helper we added in encoder_imagebind.py
        from encoder_imagebind import embed_texts as embed_fn
    else:                                   # pragma: no cover
        raise ValueError(f"unknown encoder: {args.encoder}")

    # load captions
    caps = load_coco_caps(args.json, args.limit)
    if args.verbose:
        print(f"Loaded {len(caps):,} captions from {args.json}")

    # embed in batches
    all_vecs = []
    t0 = time.time()
    for i in tqdm(range(0, len(caps), args.batch), disable=not args.verbose):
        batch = caps[i : i + args.batch]
        vec  = embed_fn(batch)              # shape (B, dim) torch.Tensor / np
        vec  = vec.detach().cpu().numpy() if hasattr(vec, "detach") else vec
        all_vecs.append(vec)
    vecs = np.concatenate(all_vecs, axis=0)
    if args.verbose:
        dt = time.time() - t0
        qps = len(caps) / dt
        print(f"✓ embedding done in {dt:,.1f}s  ({qps:,.1f} q/s)  dim={vecs.shape[1]}")

    # save
    save_embeds(args.out, vecs)


if __name__ == "__main__":
    main()
