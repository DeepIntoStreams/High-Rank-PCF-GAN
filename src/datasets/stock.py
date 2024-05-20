import torch
import pathlib
import os
from src.datasets.utils import load_data, save_data, train_test_split, rolling_window_construction
import numpy as np
import pandas as pd
import yfinance as yf
import datetime


class Stock:
    def __init__(
        self,
    ):
        self.root = pathlib.Path("data")

        data_loc = pathlib.Path("data/Stock/")

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            train_X, test_X = self._download_data()
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

    def _download_data(self):
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime(2020, 1, 1)
        Symbols = [
            'AAPL', 'LMT', 'JPM', 'AMZN', 'PG'
        ]
        result_df = pd.DataFrame()
        # iterate over each symbol
        for i in Symbols:

            # Create a ticker object
            ticker = yf.Ticker(i)

            # Fetch the historical data with 5-hour interval
            df = ticker.history(start=start, end=end, interval='1d')

            df['Mid Price_{}'.format(i)] = (df['High'] + df['Low']) / 2

            df['Log_Price_{}'.format(i)] = np.log(df['Mid Price_{}'.format(i)])
            result_df['Log_Returns_{}'.format(i)] = df['Log_Price_{}'.format(i)].diff()

        result_df = result_df.dropna()

        result_ts = torch.tensor(np.array(result_df))
        window_size = 10
        stride = 2
        rolled_ts = rolling_window_construction(result_ts, window_size, stride)

        train_X, test_X = train_test_split(rolled_ts, 0.8)