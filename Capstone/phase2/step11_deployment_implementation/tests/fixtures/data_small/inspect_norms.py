# inspect_norms.py
import numpy as np

embs = np.load("img_embs_small.npy")  
print("Embeddings shape:", embs.shape)

norms = np.linalg.norm(embs, axis=1)
for idx, n in enumerate(norms[:10]):
    print(f"{idx:3d} → norm = {n:.6f}")
print(f"min norm = {norms.min():.6f}, max norm = {norms.max():.6f}")
