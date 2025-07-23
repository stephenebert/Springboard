#!/usr/bin/env python3
# generate_coco_texts.py
# Place this in benchmark/ and run: python generate_coco_texts.py

import json
import numpy as np
import os

# Edit path to COCO captions JSON:
ANN_PATH = (
    "/Users/steph/Library/CloudStorage/OneDrive-Personal/Desktop/"
    "Springboard/Springboard/Capstone/step2/data/coco/annotations/"
    "captions_train2017.json"
)
OUT_TEXTS = "coco_caption_texts.npy"

def main():
    if not os.path.isfile(ANN_PATH):
        raise FileNotFoundError(f"Could not find annotations at {ANN_PATH}")

    print(f"Loading COCO captions from {ANN_PATH} …")
    with open(ANN_PATH, "r") as f:
        data = json.load(f)
    captions = [ann["caption"] for ann in data["annotations"]]
    print(f"Found {len(captions):,} captions. Saving to {OUT_TEXTS} …")
    np.save(OUT_TEXTS, np.array(captions, dtype=object))
    print("Done.")

if __name__ == "__main__":
    main()
