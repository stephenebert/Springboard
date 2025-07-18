"""
Image --> BLIP caption --> CLIP embed --> FAISS search --> show top-K captions.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np, faiss, argparse, textwrap, os, sys

# ---- paths to the index and caption array you just built ----
FAISS_INDEX   = "/Users/steph/Desktop/Springboard/Capstone/extra_credit/scripts/coco_caption_clip.index"
CAPTION_ARRAY = "/Users/steph/Desktop/Springboard/Capstone/extra_credit/scripts/coco_caption_texts.npy"
# -------------------------------------------------------------

print("▶ loading BLIP …")
blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()

print("▶ loading CLIP encoder …")
clip_model = SentenceTransformer("clip-ViT-B-32")

print("▶ loading FAISS index …")
index = faiss.read_index(FAISS_INDEX)
captions = np.load(CAPTION_ARRAY, allow_pickle=True)

def blip_caption(path: str) -> str:
    img = Image.open(path).convert("RGB")
    inputs = blip_proc(img, return_tensors="pt")
    ids = blip_model.generate(**inputs)
    return blip_proc.decode(ids[0], skip_special_tokens=True)

def search(caption: str, k: int = 5):
    vec = clip_model.encode([caption], convert_to_numpy=True,
                            normalize_embeddings=True).astype("float32")
    D, I = index.search(vec, k)
    return [(captions[idx], float(D[0][rank])) for rank, idx in enumerate(I[0])]

def main(img_path: str, k: int):
    print(f"\n  Image: {img_path}")
    cap = blip_caption(img_path)
    print(f"\n BLIP caption:\n   {cap}\n")

    results = search(cap, k)
    print(f"Top-{k} similar COCO captions:")
    for r, (txt, score) in enumerate(results, 1):
        wrapped = textwrap.fill(txt, width=78, subsequent_indent="      ")
        print(f"{r:>2}. ({score:.3f}) {wrapped}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to a jpg/png")
    p.add_argument("-k", type=int, default=5, help="top-K returned")
    args = p.parse_args()
    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")
    main(args.image, args.k)