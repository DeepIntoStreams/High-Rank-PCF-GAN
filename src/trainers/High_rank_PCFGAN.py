import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import AddTime, to_numpy, track_gradient_norms, track_norm, flatten_complex_matrix, toggle_grad, construct_past_dev_path, construct_future_dev_path
from src.datasets.data_preparation import XYDataset
from src.model.discriminator.path_characteristic_function import pcf
from src.model.regressor.regressor import LSTMRegressor
from src.trainers.regression_trainer import regressor_trainer
import torch.optim.swa_utils as swa_utils
import matplotlib.pyplot as plt
from os import path as pt
import os
from collections import defaultdict
import seaborn as sns
from copy import deepcopy


class HighRankPCFGANTrainer:
    def __init__(self, G, rank_1_pcf, config, embedding_layer=None, **kwargs):
        """
        Trainer class for the basic PCF-GAN, without time serier embedding module.

        Args:
            G (torch.nn.Module): PCFG generator model.
            embedding_layer (torch.nn.Module): embedding model.
            rank_1_pcf (torch.nn.Module): rank1 PCF layer.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments for the base trainer class.
        """
        super(HighRankPCFGANTrainer, self).__init__()

        self.G = G
        self.G_optimizer = torch.optim.Adam(
                self.G.parameters(), lr=config.lr_G, betas=(0, 0.9))
        # self.E = embedding_layer
        # self.E_optimizer = torch.optim.Adam(
        #     self.E.parameters(), lr=config.lr_E, betas=(0, 0.9)
        # )
        self.config = config
        self.device = config.device
        self.MC_size = config.MC_size
        self.add_time = config.add_time

        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.G_steps_per_D_step = config.G_steps_per_D_step
        self.R_steps_per_G_step = config.R_steps_per_G_step

        self.rank_1_pcf = rank_1_pcf
        self.original_regressors = []
        self.regressors_for_fake_measure = []

        self.D = pcf(num_samples=config.Rank_2_num_samples,
                     hidden_size=config.Rank_2_lie_degree,
                     input_dim=2 * self.rank_1_pcf.degree ** 2,
                     add_time=self.add_time,
                     include_initial=False)

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
        self.batch_size = config.batch_size
        self.losses_history = defaultdict(list)
        self.past_path_length = config.past_path_length
        self.future_path_length = config.future_path_length
        self.train_reg_X_dl = None
        self.tune_regression = config.fine_tune_regression


    def reset_and_fit_regressors(self, train_dl, test_dl):
        """
        Reset the set of regressors according to the number of rank 1 pcf linear maps. Fit the models and save.
        Parameters
        ----------
        train_dl: training dataset for regression module
        test_dl: test dataset for regression module

        Returns
        -------

        """
        if self.original_regressors:
            self.original_regressors = []
        else:
            for i in range(self.rank_1_pcf.num_samples):
                regressor = LSTMRegressor(
                    input_dim=self.rank_1_pcf.input_dim,
                    hidden_dim=self.config.R_hidden_dim,
                    output_dim=self.rank_1_pcf.degree,
                    n_layers=self.config.R_num_layers
                )
                regressor.to(self.device)

                if os.path.exists(self.config.exp_dir + '/trained_regressor_{}.pt'.format(i)):
                    regressor.load_state_dict(torch.load(self.config.exp_dir + '/trained_regressor_{}.pt'.format(i)))
                else:
                    R_trainer = regressor_trainer(regressor, self.config, self.device)
                    trained_regressor_X, _, loss = R_trainer.single_train(train_dl, test_dl, idx=i)
                    R_trainer.single_plot(loss, '/single_regression_test_loss_i_{}.png'.format(i))
                    torch.save(trained_regressor_X.state_dict(), self.config.exp_dir + '/trained_regressor_{}.pt'.format(i))
                regressor.eval()
                toggle_grad(regressor, False)
                self.original_regressors.append(regressor)
                cloned_regressor = deepcopy(regressor)
                self.regressors_for_fake_measure.append(cloned_regressor)

    def load_regressors(self):
        self.original_regressors = []
        for i in range(self.rank_1_pcf.num_samples):
            regressor = LSTMRegressor(
                input_dim=self.rank_1_pcf.input_dim,
                hidden_dim=self.config.R_hidden_dim,
                output_dim=self.rank_1_pcf.degree,
                n_layers=self.config.R_num_layers
            )
            regressor.to(self.device)
            regressor.load_state_dict(torch.load(self.config.exp_dir + '/trained_regressor_{}.pt'.format(i)))
            regressor.eval()
            toggle_grad(regressor, False)
            self.original_regressors.append(regressor)

    def fit(self, train_dl, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """

        self.G.to(device)
        self.D.to(device)
        for regressor in self.original_regressors:
            toggle_grad(regressor, False)
            regressor.train()
        for regressor in self.regressors_for_fake_measure:
            toggle_grad(regressor, False)
            regressor.train()

        toggle_grad(self.rank_1_pcf, False)
        self.rank_1_pcf.eval()

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(train_dl, device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, train_dl, device, step):
        """
        Performs one training step.

        Args:
            train_dl: training dataloader
            device: Device to perform training on.
            step (int): Current training step.
        """
        # for i in range(self.D_steps_per_G_step):
        #     # generate x_fake
        #
        for i in range(self.D_steps_per_G_step):
            batch_X, past_dev_X = next(iter(train_dl))

            D_loss = self.D_trainstep((batch_X, past_dev_X), device)

            if i == 0:
                self.losses_history["D_loss"].append(D_loss)

        for i in range(self.G_steps_per_D_step):
            G_loss = self.G_trainstep((batch_X, past_dev_X), device, step, i)
            self.losses_history["G_loss"].append(G_loss)
        torch.cuda.empty_cache()
        # G_loss = self.G_trainstep(x_real_batch, device, step)
        if step % 500 == 0:
            self.G_lr_scheduler.step()
            for param_group in self.G_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

    def G_trainstep(self, x_real, device, step, i=0):
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
        toggle_grad(self.D, False)
        self.G.train()
        self.G_optimizer.zero_grad()
        # self.regression_module.train()
        self.D.train()


        # Load data
        batch_X, past_dev_X = x_real
        batch_X = batch_X.to(device)
        past_dev_X = past_dev_X.to(device)

        batch_size, T, D = batch_X.shape

        # batch_X_past = batch_X[:, :self.past_path_length]
        #
        # x_fake_future = self.G(self.future_path_length, batch_X_past)
        # x_fake = torch.cat([batch_X_past, x_fake_future], dim=1) # [N, T, D]

        # On real data, use regression module to compute the expected development
        exp_dev_real = self.compute_real_dev(batch_X, past_dev_X)  # [N, m, future_length, lie_deg, lie_deg]
        # Shirnk it to [N*m, future_length, 2*lie_deg**2]
        exp_dev_real = flatten_complex_matrix(exp_dev_real).reshape(
            [batch_size, self.rank_1_pcf.num_samples, self.future_path_length, -1])

        if self.tune_regression and step % self.R_steps_per_G_step == 0 and i == 0 and step>199:
            # Before computing the fake measure, fine tune the regression model again
            self.fine_tune_regression(step)
            exp_dev_fake = self.compute_fake_dev(batch_X, track_loss=True)
            exp_dev_fake = flatten_complex_matrix(exp_dev_fake).reshape(
                [batch_size, self.rank_1_pcf.num_samples, self.future_path_length, -1])
        else:
            # For fake data, estimate the conditional expected development using MC
            exp_dev_fake = self.compute_fake_dev(batch_X)
            exp_dev_fake = flatten_complex_matrix(exp_dev_fake).reshape(
                [batch_size, self.rank_1_pcf.num_samples, self.future_path_length, -1])

        G_loss = self.D.high_rank_distance_measure(exp_dev_real, exp_dev_fake, Lambda=self.config.init_lambda)  # (T)
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
        torch.cuda.empty_cache()
        toggle_grad(self.G, False)
        if step % self.config.evaluate_every == 0 and i==0:
            self.G.eval()
            with torch.no_grad():

                batch_X_past = batch_X[:, :self.past_path_length]

                x_fake_future = self.G(self.future_path_length, batch_X_past)
                x_fake = torch.cat([batch_X_past, x_fake_future], dim=1) # [N, T, D]

            torch.save(x_fake, self.config.exp_dir + 'fake_{}.pt'.format(step))
            torch.save(self.G.state_dict(), self.config.exp_dir + 'G_{}.pt'.format(step))
            torch.save(self.D.state_dict(), self.config.exp_dir + 'D_{}.pt'.format(step))

            self.plot_sample(batch_X, x_fake, self.config, step)
            # # print(torch.stack(self.losses_history['G_loss_dyadic']).shape)
            # plt.plot(to_numpy(torch.stack(self.losses_history['G_loss_dyadic'])))
            # plt.savefig(
            #     pt.join(self.config.exp_dir, "G_loss_dyadic_" + str(step) + ".png")
            # )
            # plt.close()
            self.plot_losses(loss_item="G_loss", step=step)
            self.plot_losses(loss_item="D_loss", step=step)
            self.plot_losses(loss_item="grad_norm_G", step=step)
            self.plot_losses(loss_item="grad_norm_D", step=step)
            self.plot_losses(loss_item="norm_D", step=step)
            self.plot_losses(loss_item="norm_G", step=step)
            if step>199:
                self.plot_reg_losses(step=step)
            # self.plot_losses(loss_item="SigMMD", step=step)
            self.G.train()
        return G_loss.item()

    def D_trainstep(self, x_real, device):
        """
        Performs one training step for the discriminator.

        Args:
            x_real: Real samples for training.

        Returns:
            float: Discriminator loss value.
        """
        toggle_grad(self.D, True)
        toggle_grad(self.G, False)
        self.D.train()
        self.D_optimizer.zero_grad()

        # Load data
        batch_X, past_dev_X = x_real
        batch_X = batch_X.to(device).requires_grad_()
        past_dev_X = past_dev_X.to(device)

        batch_size, T, D = batch_X.shape
        with torch.no_grad():
            # On real data, use regression module to compute the expected development
            exp_dev_real = self.compute_real_dev(batch_X, past_dev_X)  # [N, m, future_length, lie_deg, lie_deg]
            # Shirnk it to [N*m, future_length, 2*lie_deg**2]
            exp_dev_real = flatten_complex_matrix(exp_dev_real).reshape(
                [batch_size, self.rank_1_pcf.num_samples, self.future_path_length, -1])
            # For fake data, estimate the conditional expected development using MC
            exp_dev_fake = self.compute_fake_dev(batch_X)
            exp_dev_fake = flatten_complex_matrix(exp_dev_fake).reshape(
                [batch_size, self.rank_1_pcf.num_samples, self.future_path_length, -1])
        d_loss = -self.D.high_rank_distance_measure(exp_dev_real, exp_dev_fake, Lambda=0.1)

        d_loss.backward()

        # Step discriminator params
        self.D_optimizer.step()
        torch.cuda.empty_cache()
        # Toggle gradient to False
        toggle_grad(self.D, False)

        return d_loss.item()

    def plot_losses(self, loss_item: str, step: int = 0):
        plt.plot(self.losses_history[loss_item])
        plt.savefig(
            pt.join(self.config.exp_dir, loss_item + "_" + str(step) + ".png")
        )
        plt.close()

    def plot_reg_losses(self, step: int = 0):
        for i in range(self.rank_1_pcf.num_samples):
            plt.plot(self.losses_history['reg_loss_{}'.format(i)], label='M_{}'.format(i))
        plt.legend()
        plt.savefig(
            pt.join(self.config.exp_dir, "reg_loss_" + str(step) + ".png")
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

    def compute_fake_dev_(self, x_real, x_real_past_dev=None):
        """
        Estimates the future development path under the fake measure using Monte-Carlo simulation
        Input:
        x_past: torch.Tensor, real batched paths of shape [N, T, D]
        Output:
        dev_x_fake_path: torch.Tensor expected future development path of shape [N, num_linear_maps, future_path_length+1, lie, lie]
        """
        N, T, D = x_real.shape
        # if D == self.config.data_feat_dim and self.add_time:
        #     x_real = AddTime(x_real)
        dev_x_fake_path = []

        for t in range(self.past_path_length, T):
            x_past_temp = x_real[:, t - self.past_path_length:t]
            x_past_temp_MC = x_past_temp.repeat([self.MC_size, 1, 1])
            x_future_MC = self.G(T - t, x_past_temp_MC) # Track the gradient

            # with torch.no_grad():
            x_future_dev = self.rank_1_pcf.unitary_development(
                AddTime(torch.cat((x_past_temp_MC[:, -1:], x_future_MC), dim=1))).reshape(
                [self.MC_size, N, self.rank_1_pcf.num_samples, self.rank_1_pcf.degree, self.rank_1_pcf.degree]).mean(0) # [N, m, lie_deg, lie_deg]
            x_past_dev = self.rank_1_pcf.unitary_development(AddTime(x_real[:, :t]))
            dev_x_fake = x_past_dev @ x_future_dev
            dev_x_fake_path.append(dev_x_fake)
        # Add a last development give filtration at time t=T
        # dev_x_fake = self.rank_1_pcf.unitary_development(AddTime(x_real))
        # dev_x_fake_path.append(dev_x_fake)
        dev_x_fake_path = torch.stack(dev_x_fake_path).permute([1, 2, 0, 3, 4])
        return dev_x_fake_path # [N, m, T, lie_deg, lie_deg]

    def compute_fake_dev(self, x_real, x_real_past_dev=None, track_loss=False):
        N, T, D = x_real.shape
        batch_X_past = x_real[:, :self.past_path_length]
        x_fake_future = self.G(self.future_path_length, batch_X_past)
        x_fake = torch.cat([batch_X_past, x_fake_future], dim=1)  # [N, T, D]
        if self.add_time:
            x_fake = AddTime(x_fake)
        dev_list = []
        for step in range(1, T+1):
            if step == 1:
                dev = torch.eye(self.rank_1_pcf.degree).repeat(N, self.rank_1_pcf.num_samples, 1, 1)
                dev_list.append(dev)
            else:
                dev_list.append(self.rank_1_pcf.unitary_development(x_fake[:, :step]))
        dev_list[0] = dev_list[0].to(dtype=dev_list[-1].dtype, device=dev_list[-1].device)
        x_fake_past_dev = torch.stack(dev_list).permute([1,2,0,3,4]).squeeze()

        exp_devs = []
        exp_devs_ori = []
        for i in range(self.rank_1_pcf.num_samples):
            if self.tune_regression:
                exp_dev = self.regressors_for_fake_measure[i](x_fake, self.device)
                if track_loss:
                    exp_dev_ori = self.original_regressors[i](x_fake, self.device)
                    exp_dev_ori_whole = x_fake_past_dev[:, i] @ exp_dev_ori
                    exp_devs_ori.append(exp_dev_ori_whole)
            else:
                # exp_dev = self.regressors_for_fake_measure[i](x_fake, self.device)
                exp_dev = self.original_regressors[i](x_fake, self.device)
            exp_dev_whole = x_fake_past_dev[:, i] @ exp_dev
            exp_devs.append(exp_dev_whole)
        exp_devs = torch.stack(exp_devs, dim=1)[:, :, self.past_path_length - 1:-1]
        if self.tune_regression and track_loss:
            exp_devs_ori = torch.stack(exp_devs_ori, dim=1)[:, :, self.past_path_length - 1:-1]
            for i in range(self.rank_1_pcf.num_samples):
                reg_loss = torch.norm(exp_devs[:, i] - exp_devs_ori[:, i], dim=[2, 3]).sum(1).mean().item()
                self.losses_history['reg_loss_{}'.format(i)].append(reg_loss)
        # print(exp_devs.shape)
        # return exp_devs[:, :, self.past_path_length :]
        return exp_devs


    def compute_real_dev(self, x_real, x_real_past_dev):
        """
        Estimates the future development path under the real measure using learnt regression modules
        """
        N, T, D = x_real.shape
        if D == self.config.data_feat_dim and self.add_time:
            x_real = AddTime(x_real)
        dev_x_real_path = []
        with torch.no_grad():
            exp_devs = []
            for i in range(self.rank_1_pcf.num_samples):
                exp_dev = self.original_regressors[i](x_real, self.device)
                exp_dev_whole = x_real_past_dev[:, i] @ exp_dev
                exp_devs.append(exp_dev_whole)
            exp_devs = torch.stack(exp_devs, dim=1)
        # print(exp_devs.shape)
        # Exclude the last step, not informative
        # return exp_devs[:,:,self.past_path_length:]
        return exp_devs[:, :, self.past_path_length - 1:-1]

    def x_real_dl_for_regression_training(self, x_real, batch_size=5000):
        if not self.train_reg_X_dl:
            self.train_reg_X_dl = DataLoader(x_real, batch_size, shuffle=True)

    def prepare_dl_for_regression_training(self, X_train, config):
        if self.add_time:
            X_train = AddTime(X_train)

        with torch.no_grad():
            future_dev_path_X = construct_future_dev_path(self.rank_1_pcf, X_train)

        train_reg_X_ds = XYDataset(X_train, future_dev_path_X)
        train_reg_X_dl = DataLoader(train_reg_X_ds, config.R_batch_size, shuffle=True)
        return train_reg_X_dl

    def fine_tune_regression(self, step):
        print("Fine tuning: step {}".format(step))
        toggle_grad(self.G, False)
        # Prepare data
        x_real = next(iter(self.train_reg_X_dl)).to(self.device)
        batch_X_past = x_real[:, :self.past_path_length]

        with torch.no_grad():
            x_fake_future = self.G(self.future_path_length, batch_X_past)
            x_fake = torch.cat([batch_X_past, x_fake_future], dim=1)  # [N, T, D]

        test_size = int(0.2 * x_fake.shape[0])
        x_fake_train = x_fake[:test_size]
        x_fake_test = x_fake[-test_size:]

        temp_config = deepcopy(self.config)
        temp_config.R_iterations = self.config.Finetune_R_iterations
        temp_config.R_batch_size = self.config.Finetune_R_batch_size
        temp_config.lr_R = self.config.Finetune_R_lr

        train_dl = self.prepare_dl_for_regression_training(x_fake_train, temp_config)
        test_dl = self.prepare_dl_for_regression_training(x_fake_test, temp_config)

        for i in range(self.rank_1_pcf.num_samples):
            toggle_grad(self.regressors_for_fake_measure[i], True)
            R_trainer = regressor_trainer(self.regressors_for_fake_measure[i], temp_config, self.device)
            trained_regressor_X, _, loss = R_trainer.single_train(train_dl, test_dl, idx=i, test_every=1, max_torelance=self.config.Finetune_R_max_tor)
            print(step, self.R_steps_per_G_step)
            if step % self.R_steps_per_G_step == 0:
                R_trainer.single_plot(loss, '/single_regression_test_loss_i_{}_step_{}.png'.format(i, step))
            self.regressors_for_fake_measure[i].train()
        toggle_grad(self.G, True)
        return

