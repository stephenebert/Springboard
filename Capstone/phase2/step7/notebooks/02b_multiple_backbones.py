from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch, open_clip
from open_clip import tokenize
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

# CONFIGURATION
BACKBONES   = ["RN50", "ViT-B-32", "RN101"]
PRETRAINED  = "openai"          
PRESET      = "cpu-fast"         
MAX_SAMPLES = 10000              

# COMMON PRESETS
PRESETS = {
    "gpu":      dict(device="cuda", batch=128, max_samples=None),
    "cpu-fast": dict(device="cpu",  batch=64,  max_samples=50000),
    "cpu-tiny": dict(device="cpu",  batch=16,  max_samples=10000),
}
base_cfg = PRESETS[PRESET]

# PATHS & DATA
PARQUET    = Path(r"C:/Users/steph/OneDrive/Desktop/data/metadata.parquet")
assert PARQUET.exists(), "metadata.parquet not found at specified path"
meta_all   = pd.read_parquet(PARQUET)
meta_train = meta_all[meta_all.split == "train"].reset_index(drop=True)

# IMAGE TRANSFORMS
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
])

# RUN EACH BACKBONE
for backbone in BACKBONES:
    cfg = {
        **base_cfg,
        "model": backbone,
        "pretrained": PRETRAINED,
        "max_samples": MAX_SAMPLES if MAX_SAMPLES else base_cfg["max_samples"],
        "batch": base_cfg["batch"],
    }
    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"\nRunning {backbone} on {device}, batch={cfg['batch']}, samples={cfg['max_samples']}")

    # Subsample training set
    if cfg["max_samples"] and cfg["max_samples"] < len(meta_train):
        meta = meta_train.sample(cfg["max_samples"], random_state=0).reset_index(drop=True)
    else:
        meta = meta_train.copy()
    print(f"Using {len(meta):,} samples")

    # Build DataLoader
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(meta)
        def __getitem__(self, i):
            row = meta.iloc[i]
            img = Image.open(row.image_path).convert("RGB")
            img = transform(img)
            return img, row.caption

    loader = DataLoader(
        Dataset(),
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    print(f"Prepared {len(loader)} batches")

    # Load model
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(
        backbone, pretrained=PRETRAINED, device=device
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Loaded {backbone} in {load_time:.1f}s")

    # Embed loop
    img_parts, txt_parts = [], []
    t1 = time.time()
    with torch.no_grad():
        for images, captions in tqdm(loader, desc=f"{backbone} Embedding"):
            images = images.to(device)
            tokens = tokenize(captions).to(device)
            ie = model.encode_image(images)
            ie /= ie.norm(dim=-1, keepdim=True)
            te = model.encode_text(tokens)
            te /= te.norm(dim=-1, keepdim=True)
            img_parts.append(ie.cpu().numpy())
            txt_parts.append(te.cpu().numpy())
    embed_secs = time.time() - t1
    print(f"Embedded in {embed_secs/60:.1f} minutes")

    IMG = np.concatenate(img_parts, axis=0)
    TXT = np.concatenate(txt_parts, axis=0)

    # Compute Recall@K
    def recall_at_k(sim_matrix, k):
        topk = np.argpartition(-sim_matrix, k - 1, axis=1)[:, :k]
        hits = np.any(topk == np.arange(sim_matrix.shape[0])[:, None], axis=1)
        return round(100.0 * hits.mean(), 2)

    sim = TXT @ IMG.T
    metrics = {
        f"R@{k}": recall_at_k(sim, k)
        for k in (1, 5, 10)
    }
    print("Metrics:", metrics)

    # Save artifacts
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("experiments") / f"{backbone}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "img_embs.npy", IMG)
    np.save(out_dir / "txt_embs.npy", TXT)
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    json.dump(
        {**cfg, "embed_secs": embed_secs, "param_count": sum(p.numel() for p in model.parameters())},
        open(out_dir / "config.json", "w"),
        indent=2,
    )
    print("Saved run to", out_dir)

print("\nAll backbones complete.")
"""
prints out
Metrics: {'R@1': np.float64(33.29), 'R@5': np.float64(56.26), 'R@10': np.float64(65.81)}
Saved run to experiments\RN101_20250624_214937

All backbones complete.
"""