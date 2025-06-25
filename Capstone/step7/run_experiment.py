
import argparse, json, time
from datetime import datetime
from pathlib import Path

import numpy as np, pandas as pd
from tqdm import tqdm
from numpy.linalg import norm
from PIL import Image
import torch, open_clip
from open_clip import tokenize
from torchvision import transforms as T

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="RN50",
                    help="Backbone (default = RN50)")
parser.add_argument("--preset", default="cpu-fast",
                    choices=["gpu", "cpu-fast", "cpu-tiny"])
parser.add_argument("--max", type=int, default=None,
                    help="Max samples (None = preset default)")
parser.add_argument("--batch", type=int, default=None,
                    help="Batch size override")
args = parser.parse_args()

# presets
PRESETS = {
    "gpu":      dict(device="cuda", batch=128),
    "cpu-fast": dict(device="cpu",  batch=64, max_samples=50_000),
    "cpu-tiny": dict(device="cpu",  batch=64, max_samples=10_000),
}
cfg = PRESETS[args.preset].copy()
cfg.update(dict(model=args.model,
                pretrained="laion2b_s34b_b79k",
                max_samples=args.max or PRESETS[args.preset].get("max_samples"),
                batch=args.batch or cfg["batch"]))
if cfg["device"] == "cuda" and not torch.cuda.is_available():
    cfg["device"] = "cpu"

print("Config:", cfg)

# metadata subset
PARQUET = Path(r"C:/Users/steph/OneDrive/Desktop/data/metadata.parquet")
meta = pd.read_parquet(PARQUET)
meta = meta[meta["split"] == "train"].reset_index(drop=True)
if cfg["max_samples"] and cfg["max_samples"] < len(meta):
    meta = meta.sample(cfg["max_samples"], random_state=0).reset_index(drop=True)

# dataset
DATA_ROOT = Path(r"C:/Users/steph/OneDrive/Desktop/data")
def path_from_row(r):
    if r.domain == "coco":
        return DATA_ROOT/"coco"/"train2017"/f"{int(r.image_id):012d}.jpg"
    if r.domain == "sd":
        return DATA_ROOT/"SD"/"images"/r.image_id
    return DATA_ROOT/"flickr30k"/"flickr30k_images"/r.image_id

transform = T.Compose([
    T.Resize(224, antialias=True), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.48145466,0.4578275,0.40821073),
                (0.26862954,0.26130258,0.27577711)),
])
class RetrievalDS(torch.utils.data.Dataset):
    def __len__(self): return len(meta)
    def __getitem__(self, i):
        r = meta.iloc[i]
        img = transform(Image.open(path_from_row(r)).convert("RGB"))
        return {"image": img, "text": r.caption, "id": i}
loader = torch.utils.data.DataLoader(
    RetrievalDS(), batch_size=cfg["batch"], shuffle=False,
    num_workers=4, pin_memory=(cfg["device"] == "cuda")
)

# load backbone
model, _, _ = open_clip.create_model_and_transforms(
    cfg["model"], pretrained=cfg["pretrained"], device=cfg["device"])
model.eval()
tok = tokenize
param_cnt = int(sum(p.numel() for p in model.parameters()))
print(f"{cfg['model']}  |  {param_cnt/1e6:.1f} M params")

# embed
img_emb, txt_emb = [], []
t0 = time.time()
with torch.no_grad():
    for b in tqdm(loader, desc="Embedding"):
        imgs = b["image"].to(cfg["device"], non_blocking=True)
        caps = tok(b["text"]).to(cfg["device"], non_blocking=True)
        ie = model.encode_image(imgs); ie /= ie.norm(dim=-1, keepdim=True)
        te = model.encode_text(caps); te /= te.norm(dim=-1, keepdim=True)
        img_emb.append(ie.cpu()); txt_emb.append(te.cpu())
embed_secs = round(time.time()-t0, 1)

IMG = torch.cat(img_emb).numpy().astype("float16")
TXT = torch.cat(txt_emb).numpy().astype("float16")

# recall
sim = TXT @ IMG.T
def r_at_k(mat, k): return 100* np.mean(np.any(
        np.argpartition(-mat, k-1, axis=1)[:, :k] ==
        np.arange(len(mat))[:,None], axis=1))
metrics = {f"R@{k}": round(r_at_k(sim, k), 2) for k in (1,5,10)}
print("Recall:", metrics)

# save
run_dir = Path("experiments") / f"{cfg['model']}_{datetime.now():%Y%m%d_%H%M%S}"
run_dir.mkdir(parents=True, exist_ok=True)
np.save(run_dir/"img_embs.npy", IMG)
np.save(run_dir/"txt_embs.npy", TXT)
json.dump(metrics, open(run_dir/"metrics.json","w"), indent=2)
json.dump(cfg|{\"embed_secs\": embed_secs, \"param_count\": param_cnt},
          open(run_dir/\"config.json\",\"w\"), indent=2)
print(\"Saved →\", run_dir)
