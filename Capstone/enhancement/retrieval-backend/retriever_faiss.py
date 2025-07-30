# retriever_faiss.py
import os
import faiss
import numpy as np

# ── 1) Find the index file relative to this script ───────────────────────────────
BASE_DIR = os.path.dirname(__file__)
INDEX_FILENAME = "hnsw_val2017.faiss"   # or ivfpq_val2017.faiss, etc.
INDEX_PATH = os.path.join(BASE_DIR, INDEX_FILENAME)

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH!r}")

# ── 2) Load and configure the FAISS index ────────────────────────────────────────
index = faiss.read_index(INDEX_PATH)

# For HNSW, tune the efSearch knob (higher → more accurate, slower)
# For IVF indices you could do: faiss.ParameterSpace().set_index_parameters(index, "nprobe=10")
faiss.ParameterSpace().set_index_parameters(index, "efSearch=64")

dim = index.d
print(f"Loaded FAISS index at {INDEX_PATH!r}")
print(f"  • dimension: {dim}")
print(f"  • engine:    {type(index).__name__}")
print(f"  • efSearch:  64")

# ── 3) Expose a simple search API ────────────────────────────────────────────────
def search(query_vecs: np.ndarray, top_k: int = 10):
    """
    Run a batched FAISS search.

    Args:
      query_vecs:  NumPy array of shape (N, dim), dtype float32 (or convertible).
      top_k     :  how many nearest neighbors to return.

    Returns:
      (distances, indices) each of shape (N, top_k).
    """
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype("float32")
    return index.search(query_vecs, top_k)


# ── 4) Quick smoke‐test when invoked directly ────────────────────────────────────
if __name__ == "__main__":
    # dummy vector
    x = np.random.rand(1, dim).astype("float32")
    D, I = search(x, top_k=5)
    print("Test query → distances:", D, "ids:", I)
