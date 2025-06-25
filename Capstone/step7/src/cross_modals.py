"""
cross_validate.py

Run cross-validation on your CLIP experiment:
  • Samples with multiple random seeds
  • Computes Recall@1/5/10 per seed
  • Prints per-seed and mean±std metrics
  • Shows an error-bar plot
"""

from pathlib import Path
import json, numpy as np, pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import torch, open_clip
from open_clip import tokenize
from torchvision import transforms as T

RUN_DIR     = Path("experiments/ViT-B-32_20250624_170054")
BACKBONE    = "ViT-B-32"    # e.g. "RN50", "ViT-B-32", "ViT-L-14"
PRESET      = "cpu-fast"    # "gpu", "cpu-fast", "cpu-tiny"
MAX_SAMPLES = 10000        
BATCH       = 32           
SEEDS       = [0, 1, 2, 3, 4]  # random seeds for cross-validation
DATA_ROOT   = Path(r"C:/Users/steph/OneDrive/Desktop/data")
PARQUET_P   = DATA_ROOT / "metadata.parquet"

assert PARQUET_P.exists(), "metadata.parquet not found"
meta_all = pd.read_parquet(PARQUET_P)
meta_all = meta_all[meta_all["split"]=="train"].reset_index(drop=True)

device = "cuda" if PRESET=="gpu" and torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(
    BACKBONE, pretrained="laion2b_s34b_b79k", device=device
)
model.eval()
tok = tokenize

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.48145466,0.4578275,0.40821073),
                (0.26862954,0.26130258,0.27577711)),
])

def make_loader(df):
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(df)
        def __getitem__(self, i):
            row = df.iloc[i]
            path = Path(row["image_path"])
            try:
                img = Image.open(path).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                img = Image.new("RGB", (224,224), (0,0,0))
            return transform(img), row["caption"]
    return torch.utils.data.DataLoader(
        DS(), batch_size=BATCH, shuffle=False,
        num_workers=0, pin_memory=(device=="cuda")
    )

# CROSS-VALIDATION
records = []
for seed in SEEDS:
    sub = meta_all.sample(MAX_SAMPLES, random_state=seed) \
          if MAX_SAMPLES < len(meta_all) else meta_all.copy()
    loader = make_loader(sub)

    img_feats, txt_feats = [], []
    for imgs, caps in tqdm(loader, desc=f"Seed {seed}"):
        imgs = imgs.to(device)
        caps_tok = tok(caps).to(device)
        with torch.no_grad():
            ie = model.encode_image(imgs); ie /= ie.norm(dim=-1,keepdim=True)
            te = model.encode_text(caps_tok); te /= te.norm(dim=-1,keepdim=True)
        img_feats.append(ie.cpu().numpy())
        txt_feats.append(te.cpu().numpy())

    IMG = np.concatenate(img_feats, axis=0)
    TXT = np.concatenate(txt_feats, axis=0)
    img_n = IMG / norm(IMG, axis=1, keepdims=True)
    txt_n = TXT / norm(TXT, axis=1, keepdims=True)

    def recall_k(sim, k):
        topk = np.argpartition(-sim, k-1, axis=1)[:, :k]
        return float((topk == np.arange(sim.shape[0])[:,None]).any(1).mean()*100)

    sim = txt_n @ img_n.T
    r1, r5, r10 = recall_k(sim,1), recall_k(sim,5), recall_k(sim,10)
    records.append({"seed":seed, "R@1":r1, "R@5":r5, "R@10":r10})

# AGGREGATE & PLOT
df = pd.DataFrame(records)
print("\nPer-seed:\n", df.to_string(index=False))
mean = df.mean(); std = df.std()
print("\nMean:", mean.to_dict())
print(" Std:", std.to_dict())

ks    = [1,5,10]
means = [mean[f"R@{k}"] for k in ks]
errs  = [std [f"R@{k}"] for k in ks]

plt.figure(figsize=(6,4))
plt.errorbar(ks, means, yerr=errs, fmt="-o", capsize=5)
plt.xlabel("K")
plt.ylabel("Recall (%)")
plt.title(f"Cross‐val seeds={SEEDS} on {BACKBONE} ({MAX_SAMPLES} samples)")
plt.grid(True)
plt.xticks(ks)
plt.tight_layout()
plt.show()
"""
prints
Per-seed:
  seed   R@1   R@5  R@10
    0 43.10 66.53 75.31
    1 42.29 66.25 74.88
    2 43.04 66.62 74.84
    3 42.34 65.47 74.11
    4 42.63 66.80 75.90

Mean: {'seed': 2.0, 'R@1': 42.68, 'R@5': 66.334, 'R@10': 75.008}
 Std: {'seed': 1.5811388300841898, 'R@1': 0.3795391942869665, 'R@5': 0.5222355790254055, 'R@10': 0.659143383491031}

"""