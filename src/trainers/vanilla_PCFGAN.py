import torch
from torch import nn
from tqdm import tqdm
from src.utils import AddTime, track_gradient_norms, track_norm
from src.model.discriminator.path_characteristic_function import pcf
from torch.nn.functional import one_hot
import torch.optim.swa_utils as swa_utils
import matplotlib.pyplot as plt
from os import path as pt
from src.utils import to_numpy
from functools import partial
from PIL import ImageFile
from collections import defaultdict
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class VanillaCPCFGANTrainer:
    def __init__(self, G, train_dl, config, **kwargs):
        """
        Trainer class for the basic PCF-GAN, without time serier embedding module.

        Args:
            G (torch.nn.Module): ARFNN generator model.
            train_dl (torch.utils.data.DataLoader): Training data loader.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments.
        """
        super(VanillaCPCFGANTrainer, self).__init__()

        self.G = G
        self.G_optimizer = torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9))
        self.config = config
        self.add_time = config.add_time
        self.train_dl = train_dl
        x_real = next(iter(train_dl))
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.G_steps_per_D_step = config.G_steps_per_D_step
        self.D = pcf(num_samples=config.Rank_1_num_samples,
                     hidden_size=config.Rank_1_lie_degree,
                     input_dim=x_real.shape[-1],
                     add_time=self.add_time,
                     include_initial=False)
        self.lr_D = config.lr_D
        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=config.lr_D, betas=(0, 0.9)
        )
        self.averaged_G = swa_utils.AveragedModel(G)
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=config.gamma
        )
        self.D_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.D_optimizer, gamma=config.gamma
        )
        self.n_gradient_steps = config.steps
        self.past_path_length = config.past_path_length
        self.future_path_length = config.future_path_length
        self.batch_size = x_real.shape[0]
        self.losses_history = defaultdict(list)
        self.device = config.device
        self.G_loss = 0 # Just for early stopping

    def fit(self, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """

        self.G.to(device)
        self.D.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            # if i % 2000 == 0:
            #     self.lr_D = 2*self.lr_D
            #     self.D_optimizer = torch.optim.Adam(
            #         self.D.parameters(), lr=self.lr_D, betas=(0, 0.9)
            #     )
            #     print("Lr_D updated to {}".format(self.lr_D))

            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        """
        Performs one training step.

        Args:
            device: Device to perform training on.
            step (int): Current training step.
        """
        # for i in range(self.D_steps_per_G_step):
        #     # generate x_fake
        #

        for i in range(self.D_steps_per_G_step):
            with torch.no_grad():
                x_real = next(iter(self.train_dl)).to(device)
                x_real_past = x_real[:, :self.past_path_length]

                x_fake_future = self.G(self.future_path_length, x_real_past)
                x_fake = torch.cat([x_real_past, x_fake_future], dim=1)

            D_loss = self.D_trainstep(x_real, x_fake)

            if i == 0:
                self.losses_history["D_loss"].append(D_loss)

        for i in range(self.G_steps_per_D_step):
            G_loss = self.G_trainstep(x_real, step, i)
            self.losses_history["G_loss"].append(G_loss)

        torch.cuda.empty_cache()
        self.G_loss = G_loss
        # G_loss = self.G_trainstep(x_real, device, step)
        if step % 500 == 0:
            self.G_lr_scheduler.step()
            for param_group in self.G_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

    def G_trainstep(self, x_real, step, i=0):
        """
        Performs one training step for the generator.

        Args:
            x_real: Real samples for training.
            device: Device to perform training on.
            step (int): Current training step.

        Returns:
            float: Generator loss value.
        """
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.D.train()
        x_real_past = x_real[:, :self.past_path_length]
        x_fake_future = self.G(self.future_path_length, x_real_past)
        x_fake = torch.cat([x_real_past, x_fake_future], dim=1)

        G_loss = self.D.distance_measure(x_real, x_fake, Lambda=self.config.init_lambda)  # (T)
        # print(G_loss.shape)
        # self.losses_history['G_loss_dyadic'].append(G_loss)
        # G_loss = G_loss.mean()
        G_loss.backward()
        self.losses_history["G_loss"].append(G_loss.item())

        if i == 0:
            grad_norm_G = track_gradient_norms(self.G)
            grad_norm_D = track_gradient_norms(self.D)
            norm_G = track_norm(self.G)
            norm_D = track_norm(self.D)
            self.losses_history['grad_norm_G'].append(grad_norm_G)
            self.losses_history['grad_norm_D'].append(grad_norm_D)
            self.losses_history['norm_G'].append(norm_G)
            self.losses_history['norm_D'].append(norm_D)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.G_optimizer.step()
        toggle_grad(self.G, False)
        if step % self.config.evaluate_every == 0 and i==0:
            torch.save(x_fake, self.config.exp_dir + 'fake_{}.pt'.format(step))
            torch.save(self.G.state_dict(), self.config.exp_dir + 'G_{}.pt'.format(step))
            torch.save(self.D.state_dict(), self.config.exp_dir + 'D_{}.pt'.format(step))
            self.plot_sample(x_real, x_fake, self.config, step)
            # # print(torch.stack(self.losses_history['G_loss_dyadic']).shape)
            # plt.plot(to_numpy(torch.stack(self.losses_history['G_loss_dyadic'])))
            # plt.savefig(
            #     pt.join(self.config.exp_dir, "G_loss_dyadic_" + str(step) + ".png")
            # )
            # plt.close()
            self.plot_losses(loss_item="G_loss", step=step)
            self.plot_losses(loss_item="grad_norm_G", step=step)
            self.plot_losses(loss_item="grad_norm_D", step=step)
            # self.plot_losses(loss_item="SigMMD", step=step)

        return G_loss.item()

    def D_trainstep(self, x_real, x_fake):
        """
        Performs one training step for the discriminator.

        Args:
            x_real: Real samples for training.
            x_fake: Fake samples generated by the generator.

        Returns:
            float: Discriminator loss value.
        """
        x_real.requires_grad_()
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        d_loss = -self.D.distance_measure(x_real, x_fake, Lambda=self.config.init_lambda)

        d_loss.backward()

        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return d_loss.item()

    def plot_losses(self, loss_item: str, step: int = 0):
        plt.plot(self.losses_history[loss_item])
        plt.savefig(
            pt.join(self.config.exp_dir, loss_item + "_" + str(step) + ".png")
        )
        plt.close()

    @staticmethod
    def plot_sample(real_X, fake_X, config, step):
        sns.set()

        x_real_dim = real_X.shape[-1]
        for i in range(x_real_dim):
            plt.plot(
                to_numpy(fake_X[: config.batch_size, :, i]).T, "C%s" % i, alpha=0.3
            )
        plt.savefig(pt.join(config.exp_dir, "x_fake_" + str(step) + ".png"))
        plt.close()

        for i in range(x_real_dim):
            random_indices = torch.randint(0, real_X.shape[0], (config.batch_size,))
            plt.plot(to_numpy(real_X[random_indices, :, i]).T, "C%s" % i, alpha=0.3)
        plt.savefig(pt.join(config.exp_dir, "x_real_" + str(step) + ".png"))
        plt.close()