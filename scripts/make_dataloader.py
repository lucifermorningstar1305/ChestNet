"""
@author: Adityam Ghosh
Date: 10-16-2023

"""

from typing import Optional, List, Dict, Tuple, Any, Callable
import numpy as np
import torch.utils.data as td

from sklearn.model_selection import train_test_split
from scripts.chest_dataset_preparer import ChextDataset

class NormalDataLoader(object):
    def __init__(self, normal_label: str, val_size: Optional[float]=.2, 
                 batch_sizes: Optional[Dict]={"train":32, "val":64}):

        assert val_size < 1., "Expected val_size to be in range [0, 1)"

        self.normal_label = normal_label
        self.val_size = val_size
        self.batch_sizes = batch_sizes

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

    def setup(self, data: Any, label_map: Optional[Dict]=None, resize: Optional[Tuple]=None, transformations: Optional[Dict]=None):

        if label_map is None:
            data = data.loc[data["label"] == self.normal_label]

        else:
            data["label"] = data["label"].apply(lambda x: "Normal" if x == self.normal_label else "Anomaly")

        val_data = None

        if self.val_size:
            train_data, val_data = train_test_split(data, test_size=self.val_size, random_state=32, shuffle=True)
        else:
            train_data = data


        self.train_dataset = ChextDataset(train_data, label_map, resize, transformations["train"])

        if val_data is not None:
            self.val_dataset = ChextDataset(val_data, label_map, resize, transformations["val"]) if self.val_size else None

    
    def get_train(self) -> List:

        self.train_dataloader = td.DataLoader(self.train_dataset, batch_size=self.batch_sizes["train"],  shuffle=True, num_workers=4)
        return [self.train_dataset, self.train_dataloader]
    

    
    def get_val(self) -> List:
        
        if self.val_dataset is not None:
            self.val_dataloader = td.DataLoader(self.val_dataset, batch_size=self.batch_sizes["val"],  shuffle=False, num_workers=4)
            return [self.val_dataset, self.val_dataloader]
    

class AnomalyDataLoader(object):
    def __init__(self, normal_label: str, val_size: Optional[float]=.2, 
                 batch_sizes: Optional[Dict]={"train":32, "val":64}):

        assert val_size < 1., "Expected val_size to be in range [0, 1)"

        self.normal_label = normal_label
        self.val_size = val_size
        self.batch_sizes = batch_sizes

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self._n_classes = 0

    def setup(self, data: Any, resize: Optional[Tuple]=None, transformations: Optional[Dict]=None):

        data = data.loc[data["label"] != self.normal_label]
        self._n_classes = data["label"].nunique()

        label_map = {v:k for k, v in enumerate(data["label"].unique())}

        val_data = None
        if self.val_size:
            train_data, val_data = train_test_split(data, test_size=self.val_size, random_state=32, shuffle=True)
        else:
            train_data = data

        self.train_dataset = ChextDataset(train_data, label_map, resize, transformations["train"])

        if val_data is not None:
            self.val_dataset = ChextDataset(val_data, label_map, resize, transformations["val"]) if self.val_size else None

    
    def get_train(self) -> List:

        self.train_dataloader = td.DataLoader(self.train_dataset, batch_size=self.batch_sizes["train"],  shuffle=True, num_workers=4)
        return [self.train_dataset, self.train_dataloader]
    

    
    def get_val(self) -> List:
        
        if self.val_dataset is not None:
            self.val_dataloader = td.DataLoader(self.val_dataset, batch_size=self.batch_sizes["val"],  shuffle=False, num_workers=4)
            return [self.val_dataset, self.val_dataloader]
        
    @property
    def n_classes(self):
        return self._n_classes
    
        
class TestingDataLoader(object):
    def __init__(self, label_map: Dict, batch_size: Optional[int]=32):

        self.label_map = label_map
        self.batch_size = batch_size

        self.dataset = None
        self.dataloader = None

    def setup(self, data: Any, resize: Optional[Tuple]=None, transformations: Optional[Callable]=None):

        self.dataset = ChextDataset(data, LABEL_MAP=self.label_map, resize=resize, transformations=transformations)

    def get_test(self) -> Tuple:
        
        self.dataloader = td.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return self.dataset, self.dataloader
        