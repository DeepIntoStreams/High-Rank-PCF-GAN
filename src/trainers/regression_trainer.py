import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

class regressor_trainer(nn.Module):
    def __init__(self, regressor_X, config, device):
        """
        Trainer class for the regression model
        """
        super(regressor_trainer, self).__init__()
        self.config = config
        self.regressor_X = regressor_X
        self.device = device

    def single_train(self, X_dl, X_test_dl, idx = 0, test_every = 50, max_torelance = 999):
        self.regressor_X.to(self.device)
        best_loss = 10000.
        best_test_loss = 10000.
        k = 0
        # loss = []
        # test_loss = []
        loss = {"train_loss": [], "test_loss": []}
        regressor_optimizer = torch.optim.Adam(self.regressor_X.parameters(), betas=(0, 0.9), lr=self.config.lr_R)
        self.regressor_X.train()
        for i in range(self.config.R_iterations):
            regressor_optimizer.zero_grad()
            batch_X, batch_X_dev = next(iter(X_dl))
            if len(batch_X_dev.shape) > 4:
                batch_X_dev = batch_X_dev[:, idx]
            reg_dev = self.regressor_X(batch_X, self.device)

            regressor_loss = torch.norm(reg_dev - batch_X_dev, dim=[2, 3]).sum(1).mean()
            loss["train_loss"].append(regressor_loss.item())
            if regressor_loss < best_loss:
                # print("Loss updated: {}".format(regressor_loss), " at iteration {}".format(i))
                best_loss = regressor_loss
                trained_regressor = self.regressor_X

            regressor_loss.backward()
            regressor_optimizer.step()

            if i % test_every == 0:
                with torch.no_grad():
                    self.regressor_X.eval()
                    batch_X_test, batch_X_dev_test = next(iter(X_test_dl))
                    if len(batch_X_dev_test.shape) > 4:
                        batch_X_dev_test = batch_X_dev_test[:, idx]
                    reg_dev = self.regressor_X(batch_X_test, self.device)
                    test_regressor_loss = torch.norm(reg_dev - batch_X_dev_test, dim=[2, 3]).sum(1).mean()
                    loss["test_loss"].append(test_regressor_loss.item())
                    # Track the running minimum of test loss
                    # Compare the test loss with the running minimum, if the test loss is not smaller, increase k by 1
                    if test_regressor_loss < best_test_loss:
                        best_test_loss = test_regressor_loss
                        k = 0
                    else:
                        k += 1
                    if k > max_torelance:
                        print('Overfitting detected, stop training')
                        break

                    self.regressor_X.train()
        return trained_regressor, None, loss

    def single_plot(self, loss, title=None):
        x_axis = np.linspace(0, len(loss["train_loss"]), len(loss["test_loss"]))
        plt.plot(loss["train_loss"], label="train_loss")
        plt.plot(x_axis, loss["test_loss"], label="test_loss")
        plt.legend()
        if title:
            plt.savefig(self.config.exp_dir + title)
        else:
            plt.savefig(self.config.exp_dir + '/regression_test_loss.png')
        plt.close()

class regressor_joint_trainer(regressor_trainer):
    def __init__(self, regressor_X, regressor_Y, config, device):
        """
        Trainer class for the regression model does joint training
        """
        super(regressor_joint_trainer, self).__init__(regressor_X, config, device)
        self.regressor_Y = regressor_Y

    def joint_training(self, X_dl, X_test_dl, Y_dl, Y_test_dl):
        self.regressor_X.to(self.device)
        self.regressor_Y.to(self.device)
        best_loss_X = 10000.
        best_loss_Y = 10000.
        loss = {"R1_loss": [], "R2_loss": [], "R1X_R2X": [], "R1Y_R2Y": [], "R1X_R1Y": [], "R2X_R2Y": [], "Test_R1_loss": [], "Test_R2_loss": []}
        regressor_optimizer_X = torch.optim.Adam(self.regressor_X.parameters(), betas=(0, 0.9), lr=self.config.lr_R)
        regressor_optimizer_Y = torch.optim.Adam(self.regressor_Y.parameters(), betas=(0, 0.9), lr=self.config.lr_R)
        self.regressor_X.train()
        self.regressor_Y.train()
        for i in tqdm(range(self.config.R_iterations)):
            regressor_optimizer_X.zero_grad()
            regressor_optimizer_Y.zero_grad()
            batch_X, batch_X_dev = next(iter(X_dl))
            reg_dev_X = self.regressor_X(batch_X, self.device)
            regressor_loss_X = torch.norm(reg_dev_X - batch_X_dev, dim=[2, 3]).sum(1).mean()
            loss["R1_loss"].append(regressor_loss_X.item())
            if regressor_loss_X < best_loss_X:
                print("Loss updated: {}".format(regressor_loss_X), " at iteration {}".format(i))
                #             with torch.no_grad():
                #                 print(torch.norm(reg_dev - batch_X_dev, dim = [2,3]).mean(0))
                best_loss_X = regressor_loss_X
                trained_regressor_X = self.regressor_X

            regressor_loss_X.backward()
            regressor_optimizer_X.step()

            batch_Y, batch_Y_dev = next(iter(Y_dl))
            reg_dev_Y = self.regressor_Y(batch_Y, self.device)
            regressor_loss_Y = torch.norm(reg_dev_Y - batch_Y_dev, dim=[2, 3]).sum(1).mean()
            loss["R2_loss"].append(regressor_loss_Y.item())
            if regressor_loss_Y < best_loss_Y:
                print("Loss updated: {}".format(regressor_loss_Y), " at iteration {}".format(i))
                #             with torch.no_grad():
                #                 print(torch.norm(reg_dev - batch_X_dev, dim = [2,3]).mean(0))
                best_loss_Y = regressor_loss_Y
                trained_regressor_Y = self.regressor_Y

            regressor_loss_Y.backward()
            regressor_optimizer_Y.step()

            with torch.no_grad():
                reg_dev_YX = self.regressor_Y(batch_X, self.device)
                reg_dev_XY = self.regressor_X(batch_Y, self.device)

                regressor_loss_YX = torch.norm(reg_dev_X - reg_dev_YX, dim=[2, 3]).sum(1).mean()
                regressor_loss_XY = torch.norm(reg_dev_Y - reg_dev_XY, dim=[2, 3]).sum(1).mean()

                loss["R1X_R2X"].append(regressor_loss_YX.item())
                loss["R1Y_R2Y"].append(regressor_loss_XY.item())


                regressor_loss_XXY = torch.norm(reg_dev_X - reg_dev_XY, dim=[2, 3]).sum(1).mean()
                regressor_loss_YYX = torch.norm(reg_dev_Y - reg_dev_YX, dim=[2, 3]).sum(1).mean()

                loss["R1X_R1Y"].append(regressor_loss_XXY.item())
                loss["R2X_R2Y"].append(regressor_loss_YYX.item())

            if i % 50 == 0:
                with torch.no_grad():
                    self.regressor_X.eval()
                    self.regressor_Y.eval()
                    batch_X_test, batch_X_dev_test = next(iter(X_test_dl))
                    reg_dev = self.regressor_X(batch_X_test, self.device)
                    regressor_loss = torch.norm(reg_dev - batch_X_dev_test, dim=[2, 3]).sum(1).mean()
                    loss["Test_R1_loss"].append(regressor_loss.item())
                    batch_Y_test, batch_Y_dev_test = next(iter(Y_test_dl))
                    reg_dev = self.regressor_Y(batch_Y_test, self.device)
                    regressor_loss = torch.norm(reg_dev - batch_Y_dev_test, dim=[2, 3]).sum(1).mean()
                    loss["Test_R2_loss"].append(regressor_loss.item())
                    self.regressor_X.train()
                    self.regressor_Y.train()
        # self.plot(loss)
        return trained_regressor_X, trained_regressor_Y, loss

    def plot(self, loss, title_1 = None, title_2 = None):
        plt.plot(loss["R1_loss"], label="R1_loss")
        plt.plot(loss["R2_loss"], label="R2_loss")
        plt.plot(loss["R1X_R2X"], label="R1X_R2X")
        plt.plot(loss["R1Y_R2Y"], label="R1Y_R2Y")
        plt.legend()
        if title_1:
            plt.savefig(self.config.exp_dir + title_1)
        else:
            plt.savefig(self.config.exp_dir + '/regression_loss.png')
        plt.close()

        x_axis = np.linspace(0, len(loss["R1_loss"]), len(loss["Test_R1_loss"]))
        plt.plot(loss["R1_loss"], label="R1_loss")
        plt.plot(loss["R2_loss"], label="R2_loss")
        plt.plot(x_axis, loss["Test_R1_loss"], label="Test_R1_loss")
        plt.plot(x_axis, loss["Test_R2_loss"], label="Test_R2_loss")
        plt.legend()
        if title_2:
            plt.savefig(self.config.exp_dir + title_2)
        else:
            plt.savefig(self.config.exp_dir + '/regression_test_loss.png')
        plt.close()
