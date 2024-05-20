from typing import Tuple

import torch
from torch import nn
from src.utils import init_weights
class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    # @abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """Implement here generation scheme."""
        # ...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class LSTMGenerator(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=0.1,
        BM=False,
        activation=nn.Tanh(),
    ):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.rnn.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )  # we use a simple residual network to learn the distribution at the initial time step.
        self.initial_nn1 = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )
        self.initial_nn.apply(init_weights)
        self.initial_nn1.apply(init_weights)

        self.BM = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = 0.3
        self.activation = activation

    def forward(
        self, batch_size: int, n_lags: int, device: str, z=None
    ) -> torch.Tensor:
        if z == None:
            z = (self.noise_scale * torch.randn(batch_size, n_lags, self.input_dim)).to(
                device
            )  # cumsum(1)
            if self.BM:
                z = z.cumsum(1)
            else:
                pass
            # z[:, 0, :] *= 0  # first point is fixed
            #
        else:
            z = z
        z0 = self.noise_scale * torch.randn(batch_size, self.input_dim, device=device)

        h0 = (
            self.initial_nn(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.initial_nn1(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        # c0 = torch.zeros_like(h0)

        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(self.activation(h1))

        assert x.shape[1] == n_lags

        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], flatten: bool = False):
        """
        Feedforward neural network with residual connection.

        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out


class ArFNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], latent_dim: int):
        super().__init__()
        self.network = ResFNN(input_dim+latent_dim, output_dim, hidden_dims)
        self.latent_dim = latent_dim

    def forward(self, n_lags, x_past):
        z = torch.randn(x_past.shape[0], n_lags, self.latent_dim).to(x_past.device)
        x_generated = list()
        for t in range(z.shape[1]):
            z_t = z[:, t:t + 1]
            x_in = torch.cat([z_t, x_past.reshape(x_past.shape[0], 1, -1)], dim=-1)
            x_gen = self.network(x_in)
            x_past = torch.cat([x_past[:, 1:], x_gen], dim=1)
            x_generated.append(x_gen)
        x_fake = torch.cat(x_generated, dim=1)
        return x_fake


class SimpleGenerator(ArFNN):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], latent_dim: int):
        super(SimpleGenerator, self).__init__(input_dim + latent_dim, output_dim, hidden_dims)
        self.latent_dim = latent_dim

    def sample(self, steps, x_past):
        z = torch.randn(x_past.size(0), steps, self.latent_dim).to(x_past.device)
        return self.forward(z, x_past)


class TransformerGenerator(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=0.1,
        BM=False,
        activation=nn.Tanh(),
    ):
        super(TransformerGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.transformer = nn.Transformer(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.rnn.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )  # we use a simple residual network to learn the distribution at the initial time step.
        self.initial_nn1 = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )
        self.initial_nn.apply(init_weights)
        self.initial_nn1.apply(init_weights)

        self.BM = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = 0.3
        self.activation = activation


class ConditionalLSTMGenerator(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=1.,
        BM=False,
        activation=nn.Tanh(),
    ):
        super(ConditionalLSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.latent_dim = latent_dim

        self.rnn2 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.rnn1.apply(init_weights)
        self.rnn2.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)
        self.noise_scale = noise_scale
        self.activation = activation

    def forward(self, n_lags: int, x_past) -> torch.Tensor:
        z = (self.noise_scale * torch.randn(x_past.shape[0], n_lags, self.latent_dim)).to(x_past.device)
        x_generated = list()
        x_in = x_past
        for t in range(z.shape[1]):
            z_t = z[:, t:t + 1]
            _, (h_n, c_n) = self.rnn1(x_in)
            out, _ = self.rnn2(z_t, (h_n, c_n))
            x_gen = self.linear(self.activation(out))
            x_in = torch.cat([x_in[:, 1:], x_gen], dim=1)
            x_generated.append(x_gen)
        x_fake = torch.cat(x_generated, dim=1)
        return x_fake


class ConditionalLSTMGenerator_v2(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=1.,
        BM=False,
        activation=nn.Tanh(),
    ):
        super(ConditionalLSTMGenerator_v2, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.latent_dim = latent_dim

        self.rnn1.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResFNN(latent_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )  # we use a simple residual network to learn the distribution at the initial time step.
        self.initial_nn1 = nn.Sequential(
            ResFNN(latent_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )
        self.initial_nn.apply(init_weights)
        self.initial_nn1.apply(init_weights)

        self.BM = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = 0.3

        self.activation = activation

    def forward(self, n_lags: int, x_past, z=None) -> torch.Tensor:
        batch_size = x_past.shape[0]
        device = x_past.device
        # z = (self.noise_scale * torch.randn(batch_size, n_lags, self.latent_dim)).to(device)
        if not z:
            z = (self.noise_scale * torch.randn(batch_size, n_lags, self.latent_dim)).to(device) # cumsum(1)
            if self.BM:
                z = z.cumsum(1)
            else:
                pass
            # z[:, 0, :] *= 0  # first point is fixed
            #
        else:
            z = z
        z0 = self.noise_scale * torch.randn(batch_size, self.latent_dim, device=device)

        h0 = (
            self.initial_nn(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.initial_nn1(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )

        x_past = x_past.reshape(x_past.shape[0], 1, -1)
        x_in = torch.cat((x_past.repeat(1, n_lags, 1), z), dim=-1)
        out, _= self.rnn1(x_in)
        x_fake = self.linear(self.activation(out))
        return x_fake