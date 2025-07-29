#!/usr/bin/env python3
"""
Two‑stage retrieval evaluation:

  Stage 1  – CLIP embeddings + FAISS IVF20 search (top_k candidates)
  Stage 2  – GPT‑4o selects the best candidate

Outputs Recall@1 and average latency.
"""

import os, re, json, time
from typing import List

import numpy as np
import faiss
import openai
from tqdm import tqdm
from json.decoder import JSONDecodeError

from encoder_clip import embed_texts  # stub you created earlier

# ---------- CONFIG ----------
JSON_PATH = (
    "/Users/steph/Desktop/Springboard/Capstone/step5/data/raw/"
    "captions_val2017.json"
)
LIMIT  = 4000   # how many captions/queries to evaluate  (lower => cheaper)
TOP_K  = 10    # FAISS shortlist size for GPT‑4o to rerank
# ----------------------------

def gpt4o_choice(query: str, candidates: List[str]) -> int:
    """Ask GPT‑4o to pick best caption; return zero‑based index."""
    opt_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_msg  = (
        f"Query: \"{query}\"\n\n"
        f"Choose the single best caption (reply ONLY the number):\n{opt_lines}"
    )
    for attempt in range(6):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are an expert caption selector."},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\d+", text)
            if not m:
                raise ValueError(f"No integer in GPT reply: {text!r}")
            return int(m.group()) - 1
        except (openai.error.RateLimitError, JSONDecodeError) as e:
            wait = 4 * (attempt + 1)
            print(f"[warn] {e} – retry in {wait}s ({attempt+1}/6)")
            time.sleep(wait)
    raise RuntimeError("GPT‑4o failed after 6 attempts")


def load_coco_captions(path: str, limit: int) -> List[str]:
    """Return first `limit` captions from COCO val JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    caps = [ann["caption"] for ann in data["annotations"]]
    return caps[:limit]


def main():
    # sanity check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please export OPENAI_API_KEY before running.")

    # load captions
    print(f"[1/4] Reading {JSON_PATH}")
    captions = load_coco_captions(JSON_PATH, LIMIT)
    print(f"[2/4] Loaded {len(captions)} captions – embedding CLIP vectors")

    # embed all captions once
    embeds = np.vstack([embed_texts([c]) for c in tqdm(captions)])

    # build FAISS IVF20 index
    d = embeds.shape[1]
    index = faiss.index_factory(d, "IVF20,Flat")
    index.train(embeds)
    index.add(embeds)

    # two‑stage retrieval
    hits = 0
    t0   = time.time()
    for i, q in enumerate(tqdm(captions, desc="queries")):
        q_emb = embed_texts([q])
        _, I  = index.search(q_emb, TOP_K)
        cands = [captions[j] for j in I[0]]

        best = gpt4o_choice(q, cands)
        if I[0][best] == i:
            hits += 1

    elapsed = time.time() - t0
    recall  = hits / len(captions)
    latency = elapsed / len(captions) * 1000

    print(f"\nReranked Recall@1 : {recall:6.4f}")
    print(f"Avg latency / query: {latency:6.2f} ms")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # quiet HF warning
    main()
