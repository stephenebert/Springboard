from pathlib import Path
from typing import Union, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# Default CLIP preprocessing
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

default_transform = T.Compose([
    T.Resize(224, antialias=True),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_CLIP_MEAN, _CLIP_STD),
])

class RetrievalDataset(Dataset):
    """
    PyTorch Dataset that yields a dict:
        {
          "image"  : Tensor [3,224,224],
          "caption": str,          # raw text
          "domain" : str,          # 'coco' | 'sd' | 'flickr'
          "id"     : str
        }
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        split: str = "train",
        transform: Callable = default_transform,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.df = self.df[self.df.split == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # load + preprocess image
        img = Image.open(row.image_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "caption": row.caption,  
            "domain": row.domain,
            "id": row.id,
        }
