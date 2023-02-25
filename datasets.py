from typing import Dict
from torch.utils.data import Dataset
from torch import Tensor
import torch


class SizeSquirrelRecommenderDataset(Dataset):
    def __init__(self, X_dict, y: Tensor, idxs=None, device=torch.device("cpu")):
        self.X_dict = {name: X for name, X in X_dict.items()}
        self.y = y.to(device)
        if idxs is not None:
            self.idxs = idxs
            self.X_dict, self.y = {
                name: X[idxs, :] for name, X in self.X_dict.items()
            }, self.y[idxs]
        self.len = self.y.shape[0]

    def __getitem__(self, index):
        return {name: X[index, :] for name, X in self.X_dict.items()}, self.y[index]

    def __len__(self):
        return self.len
