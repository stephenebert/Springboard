
from sentence_transformers import SentenceTransformer

# Lightweight, CPU‑efficient embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """
    texts: List[str]
    returns: np.ndarray of shape (len(texts), hidden_dim)
    """
    return model.encode(texts, convert_to_numpy=True)
