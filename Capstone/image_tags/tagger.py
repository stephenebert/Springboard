from __future__ import annotations

import datetime as _dt
import json as _json
import pathlib as _pl
import re as _re
import sys as _sys
from typing import List

import nltk
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# ─── ensure punkt + perceptron tagger are downloaded ──────────────────────────
for res, subdir in [
    ("punkt", "tokenizers"),
    ("averaged_perceptron_tagger", "taggers"),
]:
    try:
        nltk.data.find(f"{subdir}/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

# ─── where we dump the caption+tags JSON sidecars ──────────────────────────────
CAP_TAG_DIR = _pl.Path.home() / "Desktop" / "image_tags"
CAP_TAG_DIR.mkdir(exist_ok=True, parents=True)

# ─── load the BLIP model once ──────────────────────────────────────────────────
_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ─── allowed POS prefixes ──────────────────────────────────────────────────────
_POS = {"nouns": ("NN",), "adjs": ("JJ",), "verbs": ("VB",)}

def _caption_to_tags(
    caption: str,
    k: int,
    keep_nouns: bool,
    keep_adjs: bool,
    keep_verbs: bool,
) -> List[str]:
    from nltk.tokenize import wordpunct_tokenize

    allowed = []
    if keep_nouns: allowed += _POS["nouns"]
    if keep_adjs:  allowed += _POS["adjs"]
    if keep_verbs: allowed += _POS["verbs"]

    seen, out = set(), []
    for w, pos in nltk.pos_tag(wordpunct_tokenize(caption.lower())):
        if any(pos.startswith(pref) for pref in allowed):
            clean = _re.sub(r"[^a-z0-9-]", "", w)
            if clean and clean not in seen:
                out.append(clean)
                seen.add(clean)
                if len(out) >= k:
                    break
    return out

def tag_pil_image(
    img: Image.Image,
    stem: str,
    *,
    top_k: int = 5,
    keep_nouns: bool = True,
    keep_adjs: bool = True,
    keep_verbs: bool = True,
) -> List[str]:
    # 1) generate caption
    ids = _model.generate(**_processor(images=img, return_tensors="pt"), max_length=30)
    caption = _processor.decode(ids[0], skip_special_tokens=True)
    # 2) extract tags
    tags = _caption_to_tags(caption, top_k, keep_nouns, keep_adjs, keep_verbs)
    # 3) persist side-car JSON for main.py to read back
    payload = {
        "caption": caption,
        "tags": tags,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (_p := CAP_TAG_DIR / f"{stem}.json").write_text(_json.dumps(payload, indent=2))
    return tags

if __name__ == "__main__":
    if len(_sys.argv) < 2:
        _sys.exit("Usage: python tagger.py <image_path> [top_k]")
    path = _pl.Path(_sys.argv[1])
    if not path.exists():
        _sys.exit(f"File not found: {path}")
    k = int(_sys.argv[2]) if len(_sys.argv) > 2 else 5
    with Image.open(path).convert("RGB") as im:
        print("tags:", ", ".join(tag_pil_image(im, path.stem, top_k=k)))
