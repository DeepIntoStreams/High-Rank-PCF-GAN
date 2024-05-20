from functools import partial
import numpy as np
import torch
from torch import nn, nn as nn
import math
from src.model.discriminator.base import base_projection


def tridiagonal_lie_init_(tensor: torch.tensor, init_=None):
    """
    Fills in the input tensor in place with initialization on the unitary Lie algebra.

    Args:
        tensor (torch.Tensor): A multi-dimensional tensor.
        init_ (callable): Optional. A function that initializes the tensor according to some distribution.

    Raises:
        ValueError: If the tensor has less than 2 dimensions or the last two dimensions are not square.

    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError(
                "Only tensors with 2 or more dimensions are supported. "
                "Got a tensor of shape {}".format(tuple(tensor.size()))
            )

        if init_ is None:
            torch.nn.init.uniform_(tensor, -math.pi, math.pi)
        else:
            init_(tensor)

    return tensor


class unitary_tridiag(nn.Module):
    def __init__(self):
        super().__init__()

    @ staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        M, C, n = X.shape
        matrix = torch.zeros((M, C, n+1, n+1)).to(X.device).to(X.dtype)
        indices = torch.arange(0, n)
        matrix[:, :, indices, indices + 1] = X
        matrix = (matrix - torch.conj(matrix.transpose(-2, -1))) / 2
        return matrix

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        return self.frame(X)

    @ staticmethod
    def in_lie_algebra(X, eps=1e-5):
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps))


class unitary_tridiag_projection(base_projection):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """
        Projection module used to project the path increments to the Lie group path increments
        using trainable weights from the Lie algebra.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden Lie algebra matrix.
            channels (int, optional): Number of channels to produce independent Lie algebra weights. Defaults to 1.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        super(unitary_tridiag_projection, self).__init__(input_size, hidden_size, channels, init_range, **kwargs)

        A = torch.empty(
            input_size, channels, hidden_size - 1, dtype=torch.cfloat
        )
        self.A = nn.Parameter(A)
        self.reset_parameters()
        self.param_map = unitary_tridiag()

    def reset_parameters(self):
        tridiagonal_lie_init_(self.A, partial(nn.init.normal_, std=1))