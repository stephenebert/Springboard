# retriever_faiss.py
import faiss
import numpy as np

# load HNSW index
INDEX_PATH = "hnsw_val2017.faiss"          # relative to working dir
index = faiss.read_index(INDEX_PATH)

# tune search speed / recall trade‑off
faiss.ParameterSpace().set_index_parameters(index, "efSearch=64")  # or 32/128

DIM = index.d                              # embedding dimension (512)
NPROBE = getattr(index, "nprobe", None)    # HNSW ignores nprobe, IVF needs it

print(f"Loaded {INDEX_PATH} | dim={DIM} | efSearch=64")

# helper: search
def search(query_vecs: np.ndarray, top_k: int = 10):
    """
    query_vecs:  (N, dim) float32
    returns  :  (N, top_k) distances, (N, top_k) ids
    """
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype("float32")
    return index.search(query_vecs, top_k)
