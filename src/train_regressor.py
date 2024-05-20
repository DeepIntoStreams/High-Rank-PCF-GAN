import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from os import path as pt
def train_regressor(regressor, config, X_dl, X_test_dl):
    best_loss = 10000.
    loss = []
    test_loss = []
    regressor_optimizer = torch.optim.Adam(regressor.parameters(), betas=(0, 0.9), lr=0.001)
    regressor.train()
    for i in tqdm(range(config.R_iterations)):
        regressor_optimizer.zero_grad()
        batch_X, batch_X_dev = next(iter(X_dl))
        #         batch_X_dev = next(iter(Y_dl))
        # print(batch_X.shape, batch_X_dev.shape)
        reg_dev = regressor(batch_X, config.device)
        # print(reg_dev.shape, batch_X_dev.shape)
        #
        regressor_loss = torch.norm(reg_dev - batch_X_dev, dim=(2, 3)).sum(1).mean()
        loss.append(regressor_loss.item())
        if regressor_loss < best_loss:
            print("Loss updated: {}".format(regressor_loss), " at iteration {}".format(i))
            #             with torch.no_grad():
            #                 print(torch.norm(reg_dev - batch_X_dev, dim = [2,3]).mean(0))
            best_loss = regressor_loss
            trained_regressor = regressor

        regressor_loss.backward()
        regressor_optimizer.step()

        if i % 50 == 0:
            with torch.no_grad():
                regressor.eval()
                batch_X_test, batch_X_dev_test = next(iter(X_test_dl))
                reg_dev = regressor(batch_X_test, config.device)
                #         print(reg_dev.shape, batch_X_dev.shape)

                regressor_loss = torch.norm(reg_dev - batch_X_dev_test, dim=(2, 3)).sum(1).mean()
                test_loss.append(regressor_loss.item())
                regressor.train()

    return trained_regressor, loss, test_loss


def joint_training(regressor_X, regressor_Y, config, X_dl, Y_dl):
    best_loss_X = 10000.
    best_loss_Y = 10000.
    loss = {"R1_loss": [], "R2_loss": [], "R1X_R2X": [], "R1Y_R2Y": []}
    regressor_optimizer_X = torch.optim.Adam(regressor_X.parameters(), betas=(0, 0.9), lr=0.001)
    regressor_optimizer_Y = torch.optim.Adam(regressor_Y.parameters(), betas=(0, 0.9), lr=0.001)
    regressor_X.train()
    regressor_Y.train()
    for i in tqdm(range(config.iterations)):
        regressor_optimizer_X.zero_grad()
        regressor_optimizer_Y.zero_grad()
        batch_X, batch_X_dev = next(iter(X_dl))
        #         batch_X_dev = next(iter(Y_dl))
        #         print(batch_X.shape, batch_X_dev.shape)
        reg_dev_X = regressor_X(batch_X, config.device)
        #         print(reg_dev.shape, batch_X_dev.shape)

        regressor_loss_X = torch.norm(reg_dev_X - batch_X_dev, dim=(2, 3)).sum(1).mean()
        loss["R1_loss"].append(regressor_loss_X.item())
        if regressor_loss_X < best_loss_X:
            print("Loss updated: {}".format(regressor_loss_X), " at iteration {}".format(i))
            #             with torch.no_grad():
            #                 print(torch.norm(reg_dev - batch_X_dev, dim = [2,3]).mean(0))
            best_loss_X = regressor_loss_X
            trained_regressor_X = regressor_X

        regressor_loss_X.backward()
        regressor_optimizer_X.step()

        batch_Y, batch_Y_dev = next(iter(Y_dl))
        #         batch_X_dev = next(iter(Y_dl))
        #         print(batch_X.shape, batch_X_dev.shape)
        reg_dev_Y = regressor_Y(batch_Y, config.device)
        #         print(reg_dev.shape, batch_X_dev.shape)

        regressor_loss_Y = torch.norm(reg_dev_Y - batch_Y_dev, dim=(2, 3)).sum(1).mean()
        loss["R2_loss"].append(regressor_loss_Y.item())
        if regressor_loss_Y < best_loss_Y:
            print("Loss updated: {}".format(regressor_loss_Y), " at iteration {}".format(i))
            #             with torch.no_grad():
            #                 print(torch.norm(reg_dev - batch_X_dev, dim = [2,3]).mean(0))
            best_loss_Y = regressor_loss_Y
            trained_regressor_Y = regressor_Y

        regressor_loss_Y.backward()
        regressor_optimizer_Y.step()

        with torch.no_grad():
            reg_dev_YX = regressor_Y(batch_X, config.device)
            reg_dev_XY = regressor_X(batch_Y, config.device)

            regressor_loss_YX = torch.norm(reg_dev_X - reg_dev_YX, dim=(2, 3)).sum(1).mean()
            regressor_loss_XY = torch.norm(reg_dev_Y - reg_dev_XY, dim=(2, 3)).sum(1).mean()

            loss["R1X_R2X"].append(regressor_loss_YX.item())
            loss["R1Y_R2Y"].append(regressor_loss_XY.item())

    return trained_regressor_X, trained_regressor_Y, loss

def plot_reg_losses(loss, config, train_test):
    plt.plot(loss)
    plt.savefig(
        pt.join(config.exp_dir, "L2_loss_{}.png".format(train_test))
    )
    plt.close()