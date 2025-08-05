# encoder_imagebind.py
import torch
from imagebind.models import imagebind_model
from imagebind.data import load_and_transform_text
from imagebind.models.imagebind_model import ModalityType

# ------------------------------------------------------------------
# one‑time lazy‑loaded singleton
_shared_model = None
def _get_model(device="cpu"):
    global _shared_model
    if _shared_model is None:                      # first call → download weights
        _shared_model = imagebind_model.imagebind_huge(pretrained=True)
    return _shared_model.to(device).eval()
# ------------------------------------------------------------------

def embed_texts(texts):
    """
    texts: list[str]  → np.ndarray (N,1024)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _get_model(device)

    with torch.no_grad():
        txt = load_and_transform_text(texts, device)               # (N,seq,768)
        feats = model({ModalityType.TEXT: txt})[ModalityType.TEXT] # (N,1024)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy()
