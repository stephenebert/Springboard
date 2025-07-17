# scripts/patch_h5.py
import h5py

h5path = "/data/embeddings_full.h5"
with h5py.File(h5path, "r+") as f:
    # alias the two existing groups under the names the indexer expects
    if "img_embs" not in f:
        f.copy("image_embeddings", "img_embs")
    if "txt_embs" not in f:
        f.copy("text_embeddings", "txt_embs")
print("embeddings_full.h5 now has img_embs & txt_embs")
