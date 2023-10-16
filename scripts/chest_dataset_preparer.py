from typing import Any, Dict, Callable, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as td

from PIL import Image


class ChextDataset(td.Dataset):
    def __init__(self, data: Any, LABEL_MAP: Optional[Dict]=None, resize: Optional[Tuple] = None, transformations: Optional[Callable] = None):

        assert isinstance(data, pd.DataFrame), f"Expected data to be a DataFrame object. Found {type(data)}"

        self.data = data
        self.label_map = LABEL_MAP
        self.resize = resize
        self.transformations = transformations

    def __len__(self) -> int:
        return self.data.shape[0]
    

    def __getitem__(self, idx: int) -> Dict:

        img_path = self.data.iloc[idx]["img_path"]
        label = self.data.iloc[idx]["label"]

        img = Image.open(img_path).convert("L")

        if self.resize is not None:
            img = img.resize((self.resize[1], self.resize[0]), Image.Resampling.BILINEAR)
        
        if self.transformations is not None:
            img = self.transformations(img)
        
        elif self.transformations is None:
            img = np.array(img)
            img = np.expand_dims(img, axis=-1) # Adding the channel dimension: (h, w) -> (h, w, 1)
            img = np.transpose(img, (2, 0, 1)).dtype(np.float32)
            img = torch.tensor(img, dtype=torch.float)

        return {
            "img": img,
            "label": self.label_map[label] if self.label_map is not None else torch.tensor(-1, dtype=torch.long)
        }
        

