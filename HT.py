import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model.regressor.regressor import LSTMRegressor
from src.model.discriminator.path_characteristic_function import pcf
from src.utils import save_obj, load_obj
from torch.utils.data import DataLoader, Dataset
from src.datasets.fbm_ import FBM_data
from src.datasets.data_preparation import prepare_dl, transform_to_joint_dl
from src.trainers.regression_trainer import regressor_joint_trainer
from src.high_level_pcf import high_order_pcf, vanilla_pcf
import ml_collections
import seaborn as sns
import os
import yaml
from os import path as pt
import pathlib
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sns.set()
torch.manual_seed(0)
device = 'cuda'


def run_HT(config, h, i, joint_training=False, comment = "none"):

    if joint_training:
        h_1 = 0.5
        h_2 = h
    else:
        h_1 = None
        h_2 = None
    torch.manual_seed(i+52)
    device = 'cuda'
    # Construct fbm path with different Hurst parameter
    samples = 20000
    steps = 50

    bm = FBM_data(samples, dim=3, length=steps, h=0.5)
    fbm_h = FBM_data(samples, dim=3, length=steps, h=h)

    # bm_train = FBM_data(samples, dim=3, length=steps, h=0.5)
    # fbm_h_train = FBM_data(samples, dim=3, length=steps, h=h)

    bm_test = FBM_data(10000, dim=3, length=steps, h=0.5)
    fbm_h_test = FBM_data(10000, dim=3, length=steps, h=h)

    fbm_h = fbm_h.to(device)
    bm = bm.to(device)
    # fbm_h_train = fbm_h_train.to(device)
    # bm_train = bm_train.to(device)
    fbm_h_test = fbm_h_test.to(device)
    bm_test = bm_test.to(device)

    # config.R_input_dim = bm.shape[-1] + 1  # Add time
    config.data_feat_dim = bm.shape[-1]
    config.n_lags = bm.shape[2]

    train_pcf_X_dl = DataLoader(bm, config.batch_size, shuffle=True)
    test_pcf_X_dl = DataLoader(bm_test, config.batch_size, shuffle=True)

    train_pcf_Y_dl = DataLoader(fbm_h, config.batch_size, shuffle=True)
    test_pcf_Y_dl = DataLoader(fbm_h_test, config.batch_size, shuffle=True)

    # Construct rank 1 pcf discriminator
    rank_1_pcf = vanilla_pcf(config, bm.shape[-1])

    losses = rank_1_pcf.train_M(train_pcf_X_dl, train_pcf_Y_dl, test_pcf_X_dl, test_pcf_Y_dl, 500)

    save_obj(rank_1_pcf.rank_1_pcf.state_dict(), config.exp_dir + 'rank_1_pcf_i_{}_{}.pt'.format(i, config.comment))

    plt.plot(losses["Train_loss"], label='train_loss')
    plt.plot(losses["Test_loss"], label='test_loss')
    plt.legend()
    plt.savefig(config.exp_dir+'/rank_1_loss_h_{}_i_{}_{}.png'.format(h, i, config.comment))
    plt.close()

    if joint_training:
        train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl = prepare_dl(config, rank_1_pcf.rank_1_pcf, bm, bm_test, h=h_1)
        train_reg_Y_dl, test_reg_Y_dl, train_pcf_Y_dl, test_pcf_Y_dl = prepare_dl(config, rank_1_pcf.rank_1_pcf, fbm_h, fbm_h_test, h=h_2)

        train_reg_X_dl, test_reg_X_dl = transform_to_joint_dl(config, train_reg_X_dl, test_reg_X_dl, train_reg_Y_dl, test_reg_Y_dl)
    else:
        train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl = prepare_dl(config, rank_1_pcf.rank_1_pcf, bm, bm_test)
        train_reg_Y_dl, test_reg_Y_dl, train_pcf_Y_dl, test_pcf_Y_dl = prepare_dl(config, rank_1_pcf.rank_1_pcf, fbm_h, fbm_h_test)

    x_sample, _ = next(iter(train_reg_X_dl))

    regressor_for_X = LSTMRegressor(
        input_dim=x_sample.shape[-1],
        hidden_dim=config.R_hidden_dim,
        output_dim=config.Rank_1_lie_degree,
        n_layers=config.R_num_layers
    )
    regressor_for_X.to(device)

    regressor_for_Y = LSTMRegressor(
        input_dim=x_sample.shape[-1],
        hidden_dim=config.R_hidden_dim,
        output_dim=config.Rank_1_lie_degree,
        n_layers=config.R_num_layers
    )
    regressor_for_Y.to(device)

    regressor_for_Y.load_state_dict(regressor_for_X.state_dict())


    regressor_trainer = regressor_joint_trainer(regressor_for_X, regressor_for_Y, config, device)

    if joint_training:
        trained_regressor_X, trained_regressor_Y, loss = regressor_trainer.single_train(train_reg_X_dl, test_reg_X_dl)
        regressor_trainer.single_plot(loss, '/single_regression_test_loss_h_{}_i_{}_{}.png'.format(h, i, comment))
    else:
        trained_regressor_X, trained_regressor_Y, loss = regressor_trainer.joint_training(train_reg_X_dl, test_reg_X_dl,
                                                                                          train_reg_Y_dl, test_reg_Y_dl)
        regressor_trainer.plot(loss,
                               '/regression_loss_future_h_{}_i_{}_{}.png'.format(h, i, comment),
                               '/regression_test_loss_future_h_{}_i_{}_{}.png'.format(h, i, comment))


    save_obj(trained_regressor_X.state_dict(),  config.exp_dir + 'trained_regressor_X_i_{}_{}.pt'.format(i, config.comment))
    save_obj(trained_regressor_Y.state_dict(),  config.exp_dir + 'trained_regressor_Y_i_{}_{}.pt'.format(i, config.comment))

    rank_2_pcf = high_order_pcf(regressor_X=trained_regressor_X,
                                lie_degree_1=config.Rank_1_lie_degree,
                                lie_degree_2=config.Rank_2_lie_degree,
                                num_samples_2=config.Rank_2_num_samples,
                                config=config,
                                whole_dev=True,
                                regressor_Y=trained_regressor_Y,
                                add_time=True,
                                lie_group=config.lie_group,
                                device=config.device)

    losses = rank_2_pcf.train_M(train_pcf_X_dl, train_pcf_Y_dl, test_pcf_X_dl, test_pcf_Y_dl, 500)

    save_obj(rank_2_pcf.pcf_level_2.state_dict(), config.exp_dir + 'rank_2_pcf_i_{}_{}.pt'.format(i, config.comment))

    plt.plot(losses['R1X_R2Y_loss'], label="R1X_R2Y_loss")
    plt.plot(losses['R1X_R2X_loss'], label="R1X_R2X_loss")
    plt.plot(losses['R1Y_R2Y_loss'], label="R1Y_R2Y_loss")
    plt.plot(losses['R1Y_R2X_loss'], label="R1Y_R2X_loss")
    plt.plot(losses['Out-of-sample-loss'], label="Out-of-sample-loss")
    plt.legend()
    plt.savefig(config.exp_dir+'/rank_2_loss_h_{}_i_{}_{}.png'.format(h, i, config.comment))
    plt.close()

    power, type1_error, H0_stats, H1_stats = rank_2_pcf.permutation_test(test_pcf_X_dl.dataset,
                                                                         test_pcf_Y_dl.dataset,
                                                                         sample_size=200,
                                                                         num_permutations=100)

    rank_2_pcf.print_hist(H0_stats, H1_stats, 'joint_permutation_test_future_h_{}_i_{}_{}.png'.format(h, i, comment))

    return power, type1_error


if __name__ == "__main__":
    config_dir = pt.join("configs/configs_HT.yaml")
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    h_list = [0.4, 0.425, 0.45, 0.475, 0.525, 0.55, 0.575, 0.6]
    df_dict = {}
    for h in h_list:
        power_list = []
        type1_error_list = []
        config.exp_dir = "./numerical_results/HT/h_{}/".format(h)
        path = pathlib.Path(config.exp_dir)
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        for i in range(5):
            power, type1_error = run_HT(config, h, i, joint_training=False, comment=config.comment)
            power_list.append(power)
            type1_error_list.append(type1_error)
        df_dict["power_{}".format(h)] = power_list
        df_dict["typeI_error_{}".format(h)] = type1_error_list
    df = pd.DataFrame(df_dict)

    # df.to_csv("./examples/HT/{}.csv".format(config.comment))