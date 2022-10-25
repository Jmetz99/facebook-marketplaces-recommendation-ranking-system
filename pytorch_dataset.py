import pandas as pd
import torch


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        data = pd.read

    # Not dependent on index
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)