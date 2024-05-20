import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import construct_past_dev_path, construct_future_dev_path, AddTime

class XYDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.shape = X.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class XYZDataset(Dataset):
    def __init__(self, X, Y, h):
        self.X = X
        self.Y = Y
        self.h = torch.tensor(h).repeat(self.X.shape[0], 1)
        self.shape = X.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.h[index]


def prepare_dl(config, rank_1_pcf, X_train, X_test, h=None, add_time = True):
    """
    Provides dataloader for both the regression and HT testing
    """
    steps = X_train.shape[1]

    if add_time:
        X_train = AddTime(X_train)
        X_test = AddTime(X_test)

    with torch.no_grad():
        future_dev_path_X = construct_future_dev_path(rank_1_pcf, X_train)
        future_dev_path_X_test = construct_future_dev_path(rank_1_pcf, X_test)
        past_dev_path_X = construct_past_dev_path(rank_1_pcf, X_train)
        past_dev_path_X_test = construct_past_dev_path(rank_1_pcf, X_test)

    if h:
        X_train = torch.cat([X_train,
                             torch.tensor(h).repeat(X_train.shape[0],
                                                    X_train.shape[1], 1).to(device=X_train.device,
                                                                            dtype=X_train.dtype)], dim=2)
        X_test = torch.cat([X_test,
                            torch.tensor(h).repeat(X_test.shape[0],
                                                   X_test.shape[1], 1).to(device=X_train.device,
                                                                           dtype=X_train.dtype)], dim=2)

    """
    Regression dataset
    """
    train_reg_X_ds = XYDataset(X_train, future_dev_path_X)
    test_reg_X_ds = XYDataset(X_test, future_dev_path_X_test)
    train_reg_X_dl = DataLoader(train_reg_X_ds, config.batch_size, shuffle=True)
    test_reg_X_dl = DataLoader(test_reg_X_ds, config.batch_size, shuffle=True)

    """
    PCF dataset
    """
    train_pcf_X_ds = XYDataset(X_train, past_dev_path_X)
    test_pcf_X_ds = XYDataset(X_test, past_dev_path_X_test)
    train_pcf_X_dl = DataLoader(train_pcf_X_ds, config.batch_size, shuffle=True)
    test_pcf_X_dl = DataLoader(test_pcf_X_ds, config.batch_size, shuffle=True)

    return train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl


def prepare_dl_for_high_rank_pcfgan(config, rank_1_pcf, X_train, X_test, h=None, add_time = True):
    """
    Provides dataloader for both the regression and High Rank PCFGAN
    """
    steps = X_train.shape[1]

    if add_time:
        X_train_time = AddTime(X_train)
        X_test_time = AddTime(X_test)
    else:
        X_train_time = X_train
        X_test_time = X_test

    with torch.no_grad():
        future_dev_path_X = construct_future_dev_path(rank_1_pcf, X_train_time)
        future_dev_path_X_test = construct_future_dev_path(rank_1_pcf, X_test_time)
        past_dev_path_X = construct_past_dev_path(rank_1_pcf, X_train_time)
        past_dev_path_X_test = construct_past_dev_path(rank_1_pcf, X_test_time)

    if h:
        X_train_time = torch.cat([X_train_time,
                                  torch.tensor(h).repeat(X_train_time.shape[0],
                                                         X_train_time.shape[1], 1).to(device=X_train_time.device,
                                                                                      dtype=X_train_time.dtype)], dim=2)
        X_test_time = torch.cat([X_test_time,
                                 torch.tensor(h).repeat(X_test_time.shape[0],
                                                        X_test_time.shape[1], 1).to(device=X_train_time.device,
                                                                                    dtype=X_train_time.dtype)], dim=2)

    """
    Regression dataset
    """
    train_reg_X_ds = XYDataset(X_train_time, future_dev_path_X)
    test_reg_X_ds = XYDataset(X_test_time, future_dev_path_X_test)
    train_reg_X_dl = DataLoader(train_reg_X_ds, config.batch_size, shuffle=True)
    test_reg_X_dl = DataLoader(test_reg_X_ds, config.batch_size, shuffle=True)

    """
    PCF dataset
    """
    train_pcf_X_ds = XYDataset(X_train, past_dev_path_X)
    test_pcf_X_ds = XYDataset(X_test, past_dev_path_X_test)
    train_pcf_X_dl = DataLoader(train_pcf_X_ds, config.batch_size, shuffle=True)
    test_pcf_X_dl = DataLoader(test_pcf_X_ds, config.batch_size, shuffle=True)

    return train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl

def transform_to_joint_dl(config, train_reg_X_dl, test_reg_X_dl, train_reg_Y_dl, test_reg_Y_dl):
    X_train, future_dev_path_X = train_reg_X_dl.dataset.X, train_reg_X_dl.dataset.Y
    X_test, future_dev_path_X_test = test_reg_X_dl.dataset.X, test_reg_X_dl.dataset.Y

    Y_train, future_dev_path_Y = train_reg_Y_dl.dataset.X,  train_reg_Y_dl.dataset.Y
    Y_test, future_dev_path_Y_test = test_reg_Y_dl.dataset.X, test_reg_Y_dl.dataset.Y

    train_reg_X_ds = XYDataset(torch.cat([X_train, Y_train]), torch.cat([future_dev_path_X, future_dev_path_Y]))
    test_reg_X_ds = XYDataset(torch.cat([X_test, Y_test]), torch.cat([future_dev_path_X_test, future_dev_path_Y_test]))

    train_reg_dl = DataLoader(train_reg_X_ds, config.batch_size, shuffle=True)
    test_reg_dl = DataLoader(test_reg_X_ds, config.batch_size, shuffle=True)

    return train_reg_dl, test_reg_dl


def prepare_dl_with_h(config, rank_1_pcf, X_train, X_test, h, add_time = True):
    """
    Provides dataloader for both the regression and HT testing
    """
    steps = X_train.shape[1]

    if add_time:
        X_train = AddTime(X_train)
        X_test = AddTime(X_test)
    print(steps)
    with torch.no_grad():
        future_dev_path_X = construct_future_dev_path(rank_1_pcf, X_train, steps)
        future_dev_path_X_test = construct_future_dev_path(rank_1_pcf, X_test, steps)
        past_dev_path_X = construct_past_dev_path(rank_1_pcf, X_train, steps)
        past_dev_path_X_test = construct_past_dev_path(rank_1_pcf, X_test, steps)

    """
    Regression dataset
    """
    train_reg_X_ds = XYZDataset(X_train, future_dev_path_X, h)
    test_reg_X_ds = XYZDataset(X_test, future_dev_path_X_test, h)
    train_reg_X_dl = DataLoader(train_reg_X_ds, config.batch_size, shuffle=True)
    test_reg_X_dl = DataLoader(test_reg_X_ds, config.batch_size, shuffle=True)

    """
    PCF dataset
    """
    train_pcf_X_ds = XYZDataset(X_train, past_dev_path_X, h)
    test_pcf_X_ds = XYZDataset(X_test, past_dev_path_X_test, h)
    train_pcf_X_dl = DataLoader(train_pcf_X_ds, config.batch_size, shuffle=True)
    test_pcf_X_dl = DataLoader(test_pcf_X_ds, config.batch_size, shuffle=True)

    return train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl