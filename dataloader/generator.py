import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class StockDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.Y)


def collate_fn(batch):
    x, y = zip(*batch)
    x = np.array(x)
    y = np.array(y)

    x = torch.tensor(x).to(dtype=torch.float32)
    y = torch.tensor(y).to(dtype=torch.float32)
    return x, y


def get_set_and_loader(X, Y, batch_size=64, shuffle=False):
    dataset = StockDataset(X=X, Y=Y)

    if batch_size == 0:
        batch_size = len(dataset)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn)

    return dataset, loader
