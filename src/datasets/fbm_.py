from fbm import FBM
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split, rolling_window_construction
import numpy as np
import pandas as pd

def FBM_dl(num_samples, dim, length, hursts: float, batch_size: int,
           scale: float):
    fbm_paths = []
    for i in range(num_samples*dim):
        f = FBM(n=length, hurst=hursts, method='daviesharte')
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(fbm_paths).reshape(
        num_samples, dim, length).permute(0, 2, 1)

    data = scale*data[:, 1:]
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)


def FBM_data(num_samples, dim, length, h):
    fbm_paths = []
    for i in range(num_samples*dim):
        f = FBM(n=length, hurst=h, method='daviesharte')
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(np.array(fbm_paths)).reshape(
        num_samples, dim, length+1).permute(0, 2, 1)
    return data


class fbm:
    def __init__(
        self,
    ):
        self.root = pathlib.Path("data")

        data_loc = pathlib.Path("data/fBM/")

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            train_X, test_X = self.sample_data()
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )

    @staticmethod
    def load_data(data_loc, partition):
        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
        elif partition == "test":
            X = tensors["test_X"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))

        return X

    def sample_data(self):
        fbm_train = FBM_data(20000, dim=3, length=10, h=0.25)
        fbm_test = FBM_data(20000, dim=3, length=10, h=0.25)
        return fbm_train, fbm_test