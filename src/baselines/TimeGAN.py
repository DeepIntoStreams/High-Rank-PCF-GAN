import torch
from torch.autograd import Variable
import functools

import torch
from torch import autograd

from src.baselines.base import BaseTrainer
from tqdm import tqdm
from src.utils import sample_indices, AddTime
from torch.nn.functional import one_hot
import torch.nn.functional as F
import torch.nn as nn
from src.utils import init_weights
from collections import defaultdict
import torch.optim.swa_utils as swa_utils


class TimeGAN_module(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, out_dim, activation=None):
        super(TimeGAN_module, self).__init__()
        self.input_dim = input_dim

        self.model = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                             num_layers=n_layers, batch_first=True, bidirectional=False)
        #self.linear = nn.Linear(hidden_dim*2, out_dim, bias=False)
        self.model.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, out_dim)

        self.linear.apply(init_weights)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.model(x)[0]

        x = self.linear(h)
        if self.activation == None:
            return x
        else:
            return self.activation(x)


class TIMEGANTrainer(BaseTrainer):
    def __init__(self, G, gamma, train_dl, config,
                 **kwargs):
        super(TIMEGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = TimeGAN_module(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim, out_dim=1, n_layers=config.D_num_layers).to(config.device)
        self.embedder = TimeGAN_module(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim, out_dim=config.input_dim, n_layers=config.D_num_layers, activation=nn.Sigmoid()).to(config.device)
        self.recovery = TimeGAN_module(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim, out_dim=config.input_dim, n_layers=config.D_num_layers).to(config.device)
        self.supervisor = TimeGAN_module(
            input_dim=config.input_dim, hidden_dim=config.D_hidden_dim, out_dim=config.input_dim, n_layers=config.D_num_layers, activation=nn.Sigmoid()).to(config.device)
        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.embedder_optimizer = torch.optim.Adam(
            self.embedder.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.recovery_optimizer = torch.optim.Adam(
            self.recovery.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.supervisor_optimizer = torch.optim.Adam(
            self.supervisor.parameters(), lr=config.lr_D, betas=(0, 0.9))  # Using TTUR
        self.averaged_G = swa_utils.AveragedModel(G)
        self.gamma = gamma
        self.train_dl = train_dl
        x_real = next(iter(train_dl))
        self.reg_param = 0
        self.losses_history = defaultdict(list)
        self.past_path_length = config.past_path_length
        self.future_path_length = config.future_path_length
        self.batch_size = x_real.shape[0]

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)
        self.supervisor.to(device)
        self.embedder.to(device)
        self.recovery.to(device)
        self.train_Embedder(device)
        self.train_supervisor(device)
        self.joint_train(device)

    def train_Embedder(self, device):
        toggle_grad(self.embedder, True)
        toggle_grad(self.recovery, True)
        for i in tqdm(range(self.n_gradient_steps)):

            x_real = next(iter(self.train_dl)).to(device)
            x_real.requires_grad_()
            x_real_future = x_real[:, self.past_path_length:]
            H = self.embedder(x_real_future)
            X_tilde = self.recovery(H)
            E_loss_T0 = nn.MSELoss()(x_real_future, X_tilde)
            E_loss0 = 10*torch.sqrt(E_loss_T0)
            self.embedder_optimizer.zero_grad()
            self.recovery_optimizer.zero_grad()
            E_loss0.backward(retain_graph=True)
            self.embedder_optimizer.step()
            self.recovery_optimizer.step()
        toggle_grad(self.embedder, False)
        toggle_grad(self.recovery, False)

    def train_supervisor(self, device):

        toggle_grad(self.G, True)
        toggle_grad(self.supervisor, True)
        self.G.train()
        self.supervisor.train()
        for i in tqdm(range(self.n_gradient_steps)):

            x_real = next(iter(self.train_dl)).to(device)
            x_real.requires_grad_()
            x_real_future = x_real[:, self.past_path_length:]
            H = self.embedder(x_real_future)
            # E_hat = self.G(batch_size=self.batch_size,
            #                n_lags=self.config.n_lags, device=device)

            x_real_past = x_real[:, :self.past_path_length]
            # x_fake_future = self.G(self.future_path_length, x_real_past)
            E_hat = self.G(self.future_path_length, x_real_past)
            # E_hat = torch.cat([x_real_past, x_fake_future], dim=1)
            H_hat_supervise = self.supervisor(H)
            G_loss_S = nn.MSELoss()(
                H[:, 1:, :], H_hat_supervise[:, :-1, :])
            self.G_optimizer.zero_grad()
            self.supervisor_optimizer.zero_grad()
            G_loss_S.backward(retain_graph=True)
            self.G_optimizer.step()
            self.supervisor_optimizer.step()
        toggle_grad(self.G, False)
        toggle_grad(self.supervisor, False)

    def joint_train(self, device):
        self.G.train()
        self.D.train()
        self.supervisor.train()
        self.embedder.train()
        self.recovery.train()
        for i in tqdm(range(self.n_gradient_steps)):
            for kk in range(2):
                toggle_grad(self.G, True)
                toggle_grad(self.supervisor, True)
                # generator
                x_real = next(iter(self.train_dl)).to(device)
                x_real.requires_grad_()
                x_real_future = x_real[:, self.past_path_length:]
                H = self.embedder(x_real_future)
                X_tilde = self.recovery(H)
                # E_hat = self.G(batch_size=self.batch_size,
                #                n_lags=self.config.n_lags, device=device)
                x_real_past = x_real[:, :self.past_path_length]
                # x_fake_future = self.G(self.future_path_length, x_real_past)
                E_hat = self.G(self.future_path_length, x_real_past)
                # E_hat = torch.cat([x_real_past, x_fake_future], dim=1)
                H_hat = self.supervisor(E_hat)
                H_hat_supervise = self.supervisor(H)

                X_hat = self.recovery(H_hat)
                X_hat = torch.cat([x_real_past, X_hat], dim=1)
                # Discriminator
                Y_fake = self.D(H_hat)
                Y_fake_e = self.D(E_hat)

                # Generator loss
                # 1. Adversarial loss
                G_loss_U = self.compute_loss(Y_fake, 1.)
                G_loss_U_e = self.compute_loss(Y_fake_e, 1.)

                # 2. Supervised loss
                G_loss_S = nn.MSELoss()(
                    H[:, 1:, :], H_hat_supervise[:, 1:, :])

                # 3. Two Momments
                G_loss_V1 = torch.mean(torch.abs(
                    (torch.std(X_hat, [0], unbiased=False)) + 1e-6 - (torch.std(x_real, [0]) + 1e-6)))
                G_loss_V2 = torch.mean(
                    torch.abs((torch.mean(X_hat, [0]) - (torch.mean(x_real, [0])))))

                G_loss_V = G_loss_V1 + G_loss_V2

                # 4. Summation
                G_loss = G_loss_U + self.gamma * G_loss_U_e + \
                    100 * torch.sqrt(G_loss_S) + 100*G_loss_V

                self.G_optimizer.zero_grad()
                self.supervisor_optimizer.zero_grad()

                G_loss.backward()
                self.G_optimizer.step()
                self.supervisor_optimizer.step()
                toggle_grad(self.G, False)
                toggle_grad(self.supervisor, False)
                toggle_grad(self.embedder, True)
                toggle_grad(self.recovery, True)
                H = self.embedder(x_real)
                X_tilde = self.recovery(H)
                H_hat_supervise = self.supervisor(H)
                G_loss_S = nn.MSELoss()(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Embedder network loss
                E_loss_T0 = nn.MSELoss()(x_real, X_tilde)
                E_loss0 = 10*torch.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1*G_loss_S

                self.embedder_optimizer.zero_grad()
                self.recovery_optimizer.zero_grad()

                E_loss.backward()
                self.embedder_optimizer.step()
                self.recovery_optimizer.step()
                toggle_grad(self.embedder, False)
                toggle_grad(self.recovery, False)
            # discriminator
            toggle_grad(self.D, True)
            x_real = next(iter(self.train_dl)).to(device)
            # E_hat = self.G(batch_size=self.batch_size,
            #                n_lags=self.config.n_lags, device=device)
            x_real_past = x_real[:, :self.past_path_length]
            # x_fake_future = self.G(self.future_path_length, x_real_past)
            E_hat = self.G(self.future_path_length, x_real_past)
            # E_hat = torch.cat([x_real_past, x_fake_future], dim=1)
            H_hat = self.supervisor(E_hat)
            H = self.embedder(x_real)

            # Discriminator
            Y_fake = self.D(H_hat)
            Y_real = self.D(H)
            Y_fake_e = self.D(E_hat)

            # On real data

            # On fake data

            D_loss_real = self.compute_loss(Y_real, 1.)
            D_loss_fake = self.compute_loss(Y_fake, 0.)
            D_loss_fake_e = self.compute_loss(Y_fake_e, 0.)
            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            self.D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            # Step discriminator params
            self.D_optimizer.step()
            toggle_grad(self.D, False)
            if i % self.config.evaluate_every == 0:
                # self.evaluate(X_hat, x_real, i, self.config)
                self.plot_sample(x_real, X_hat, self.config, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
