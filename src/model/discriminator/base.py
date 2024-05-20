from functools import partial
import numpy as np
import torch
from torch import nn, nn as nn
import math
from src.model.discriminator.utils import *

class base_projection(nn.Module):
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
        super(base_projection, self).__init__()
        self.__dict__.update(kwargs)
        A = torch.empty(
            input_size, channels, hidden_size, hidden_size, dtype=torch.cfloat
        )
        self.A = nn.Parameter(A)
        self.channels = channels
        self.hidden_size = hidden_size

        self.param_map = None
        self.triv = torch.linalg.matrix_exp
        self.init_range = init_range
        self.reset_parameters()

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection module.

        Args:
            dX (torch.Tensor): Tensor of shape (N, input_size).

        Returns:
            torch.Tensor: Tensor of shape (N, channels, hidden_size, hidden_size).
        """
        A = self.param_map(self.A).permute(1, 2, -1, 0)  # C,m,m,in
        AX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        return rescaled_matrix_exp(self.triv, AX)
