import numpy as np
import torch
from torch import nn
from src.model.discriminator.tridiagonal_unitary import unitary_tridiag_projection
from src.model.discriminator.unitary import unitary_projection

class development_layer(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            lie_group: str = 'unitary',
            channels: int = 1,
            include_initial: bool = False,
            partition_size=0,
            init_range=1,
    ):
        """
        Development layer module used for computation of unitary feature on time series.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden matrix.
            channels (int, optional): Number of channels. Defaults to 1.
            include_inital (bool, optional): Whether to include the initial value in the input. Defaults to False.
            return_sequence (bool, optional): Whether to return the entire sequence or just the final output. Defaults to False.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        super(development_layer, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        if lie_group == 'unitary':
            self.projection = unitary_projection(
                input_size, hidden_size, channels, init_range=init_range
            )

        elif lie_group == 'unitary_tridiag':
            self.projection = unitary_tridiag_projection(input_size, hidden_size, channels, init_range)
        else:
            raise ValueError("Please provide a valid lie group.")
        self.complex = True
        self.include_initial = include_initial
        self.partition_size = partition_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the development layer module.

        Args:
            input (torch.Tensor): Tensor with shape (N, T, input_size).

        Returns:
            torch.Tensor: Tensor with shape (N, C, hidden_size, hidden_size).
        """
        if self.complex:
            input = input.cfloat()

        N, T, C = input.shape
        if self.include_initial:
            input = torch.cat([torch.zeros((N, 1, C)).to(input.device), input], dim=1)

        dX = input[:, 1:] - input[:, :-1]
        # N,T-1,input_size

        M_dX = self.projection(dX.reshape(-1, dX.shape[-1])).reshape(
            N, -1, self.channels, self.hidden_size, self.hidden_size
        )

        if self.partition_size:
            return self.up_dyadic_prod(M_dX) # [N, 2**n, C, m, m]
        else:
            return self.dyadic_prod(M_dX)

    @staticmethod
    def dyadic_prod(X: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative product on matrix time series with dyadic partitioning.

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, C, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)

        return X[:, 0]


    def up_dyadic_prod(self, X: torch.Tensor):
        """
        Computes the cumulative product on matrix time series with dyadic partitioning.

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, 2**n, C, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        # print("MAX level: ", max_level, self.partition_size)
        # If partition_size is provided, then the whole interval is divided into subintervals of length 2**n
        # if self.partition_size:
        #     max_level = min(max_level, self.partition_size)
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        dyadic_dev = 0
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)
            # If partition_size is provided, then the whole interval is divided into subintervals of length 2**n, track the dyadic dev
            if self.partition_size and i == self.partition_size:
                dyadic_dev = X.clone()
        return X, dyadic_dev