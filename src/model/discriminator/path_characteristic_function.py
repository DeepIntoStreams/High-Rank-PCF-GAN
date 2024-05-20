import torch
import torch.nn as nn
from src.model.discriminator.development import development_layer
from src.utils import AddTime, AddTime_4d

class pcf(nn.Module):
    def __init__(self,
                 num_samples,
                 hidden_size,
                 input_dim,
                 add_time: bool,
                 lie_group = 'unitary',
                 partition_size: int = 0,
                 init_range: float = 1,
                 include_initial: bool = False):
        """
        Path characteristic function class from paths
        Args:
            num_samples: the number of linear maps L(R^d, u(n))
            hidden_size: the degree of the unitary Lie algebra
            input_dim: the path dimension, R^d
            add_time: Apply time augmentation
        """
        super(pcf, self).__init__()
        self.num_samples = num_samples
        self.degree = hidden_size
        self.input_dim = input_dim
        self.lie_group = lie_group
        self.partition_size = partition_size
        if add_time:
            self.input_dim = input_dim + 1
        else:
            self.input_dim = input_dim + 0
        self.unitary_development = development_layer(input_size=self.input_dim,
                                                     hidden_size=self.degree,
                                                     lie_group=lie_group,
                                                     channels=self.num_samples,
                                                     include_initial=include_initial,
                                                     partition_size=partition_size,
                                                     init_range=init_range)

        for param in self.unitary_development.parameters():
            param.requires_grad = True
        self.add_time = add_time

    def reset_parameters(self):
        pass

    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor, keep_time_dim=False):
        """
        Hilbert-Schmidt norm computation.

        Args:
            X (torch.Tensor): Complex-valued tensor of shape (C, m, m) or (C, T, m, m).
            Y (torch.Tensor): Tensor of the same shape as X.

        Returns:
            torch.float: Hilbert-Schmidt norm of X and Y.
        """
        assert len(X.shape) == len(Y.shape), "The dimension of X and Y must agree."
        # print(X.shape)
        if len(X.shape) == 4:
            # print(keep_time_dim)
            D = torch.einsum("bcij,bcjk->bcik", X, torch.conj(Y).permute(0, 1, 3, 2))
            # return (torch.einsum("bcii->bc", D)).mean().real
            if keep_time_dim:
                # print((torch.einsum("bcii->bc", D)))
                return (torch.einsum("bcii->bc", D)).mean(-1).real
            else:
                return (torch.einsum("bcii->bc", D)).mean().real
        elif len(X.shape) == 3:
            D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
            return (torch.einsum("bii->b", D)).mean().real
        else:
            raise ValueError("The dimension of X must be either 3 or 4.")

    def distance_measure(
            self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1, keep_time_dim=False
    ) -> torch.float:
        """
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """

        if self.partition_size:
            return self.dyadic_distance_measure(X1, X2, Lambda, keep_time_dim)
        # print(X1.shape)
        if self.add_time:
            X1 = AddTime(X1)
            X2 = AddTime(X2)
        else:
            pass
        # print(X1.shape)
        dev1, dev2 = self.unitary_development(X1), self.unitary_development(X2)
        N, T, d = X1.shape

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(0), dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_CF_1 = self.unitary_development(initial_incre_X1).mean(0)
            initial_CF_2 = self.unitary_development(initial_incre_X2).mean(0)
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + Lambda * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2, keep_time_dim=keep_time_dim
            )
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim)

    def dyadic_distance_measure(
        self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1, keep_time_dim = False
    ) -> torch.float:
        """
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """
        # print(X1.shape)
        if self.add_time:
            X1 = AddTime(X1)
            X2 = AddTime(X2)
        else:
            pass
        dev1, dyadic_dev1 = self.unitary_development(X1)
        dev2, dyadic_dev2 = self.unitary_development(X2)
        N, T, d = X1.shape
        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(0), dev2.mean(0)
        dyadic_CF1, dyadic_CF2 = dyadic_dev1.mean(0), dyadic_dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_dev_1, _ = self.unitary_development(initial_incre_X1)
            initial_dev_2, _ = self.unitary_development(initial_incre_X2)

            initial_CF_1 = initial_dev_1.mean(0)
            initial_CF_2 = initial_dev_2.mean(0)

            # initial_dyadic_CF_1, initial_dyadic_CF_2 = initial_dyadic_dev_1.mean(0), initial_dyadic_dev_2.mean(0)

            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + \
                Lambda * self.HS_norm(initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2, keep_time_dim=keep_time_dim) + \
                self.HS_norm(dyadic_CF1 - dyadic_CF2, dyadic_CF1 - dyadic_CF2, keep_time_dim=keep_time_dim)
                # Lambda * self.HS_norm(initial_dyadic_CF_1 - initial_dyadic_CF_2, initial_dyadic_CF_1 - initial_dyadic_CF_2, keep_time_dim=keep_time_dim)
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + self.HS_norm(dyadic_CF1 - dyadic_CF2, dyadic_CF1 - dyadic_CF2, keep_time_dim=keep_time_dim)

    def high_rank_distance_measure(
            self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1, keep_time_dim=False
    ) -> torch.float:
        """
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Development path samples with shape (N_1, C, T, d).
            X2 (torch.tensor): Development path samples with shape (N_2, C, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
              Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """

        if self.add_time:
            X1 = AddTime_4d(X1)
            X2 = AddTime_4d(X2)
        else:
            pass

        N, C1, T, d = X1.shape

        dev1, dev2 = self.unitary_development(X1.reshape([N*C1, T, d])), self.unitary_development(X2.reshape([N*C1, T, d]))

        M, C2, lie, _= dev1.shape

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.reshape([N, C1, C2, lie, lie]).mean(0), dev2.reshape([N, C1, C2, lie, lie]).mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, C1, 1, d)).to(X1.device), X1[:, :, 0, :].unsqueeze(2)], dim=2
            ).reshape([N*C1, -1, d])
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, C1, 1, d)).to(X1.device), X2[:, :, 0, :].unsqueeze(2)], dim=2
            ).reshape([N*C1, -1, d])
            initial_CF_1 = self.unitary_development(initial_incre_X1).reshape([N, C1, C2, lie, lie]).mean(0)
            initial_CF_2 = self.unitary_development(initial_incre_X2).reshape([N, C1, C2, lie, lie]).mean(0)
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim) + Lambda * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2, keep_time_dim=keep_time_dim
            )
        else:
            return self.HS_norm(CF1 - CF2, CF1 - CF2, keep_time_dim=keep_time_dim)