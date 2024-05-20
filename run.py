from torch.utils.data import DataLoader
from src.trainers.vanilla_PCFGAN import VanillaCPCFGANTrainer
import torch
import numpy as np
from src.model.discriminator.path_characteristic_function import pcf
from src.datasets.data_preparation import prepare_dl_for_high_rank_pcfgan
from src.model.generator.generator import ArFNN, ConditionalLSTMGenerator
from src.trainers.High_rank_PCFGAN import HighRankPCFGANTrainer
from src.evaluations.summary import full_evaluation_latest
from src.utils import toggle_grad
import ml_collections
import seaborn as sns
from os import path as pt
import yaml
from src.utils import load_obj, save_obj
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sns.set()
torch.manual_seed(0)
device = 'cuda'

def main(dataset):
    """
    Main function for training a synthetic data generator.

    Args:
        args (object): Configuration object containing the experiment settings.

    Returns:
        tuple: A tuple containing the discriminative score, predictive score, and signature MMD.
    """

    """
    ----------------------- Vanilla PCFGAN training -----------------------
    """

    config_dir = pt.join("configs/vanillaCPCFGAN/{}.yaml".format(dataset))
    with open(config_dir) as file:
        rank_1_config = ml_collections.ConfigDict(yaml.safe_load(file))

    # print(config)
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = rank_1_config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(rank_1_config)
    if rank_1_config.device == "cuda" and torch.cuda.is_available():
        rank_1_config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        rank_1_config.update({"device": "cpu"}, allow_val_change=True)

    rank_1_config.exp_dir = './numerical_results/{}/grid_algo_{}_comment_{}/'.format(
        rank_1_config.dataset,
        rank_1_config.gan_algo,
        rank_1_config.comment)
    rank_1_config.data_dir = './data/{}/'.format(rank_1_config.dataset)

    # Vanilla PCFGAN training
    train_X = load_obj(rank_1_config.data_dir + 'train_X.pt').float().to(rank_1_config.device)
    train_X_dl = DataLoader(train_X, rank_1_config.batch_size, shuffle=True)

    rank_1_config.n_lags = train_X.shape[1]
    rank_1_config.data_feat_dim = train_X.shape[-1]

    os.makedirs(rank_1_config.exp_dir, exist_ok=True)
    save_obj(rank_1_config, rank_1_config.exp_dir + '/config.pkl')

    generator = ConditionalLSTMGenerator(input_dim=rank_1_config.data_feat_dim,
                                         output_dim=rank_1_config.data_feat_dim,
                                         hidden_dim=rank_1_config.G_hidden_dim,
                                         latent_dim=rank_1_config.G_latent_dim,
                                         noise_scale=rank_1_config.noise_scale,
                                         n_layers=2)

    cpcfgan_trainer = VanillaCPCFGANTrainer(generator, train_X_dl, rank_1_config)
    cpcfgan_trainer.fit(rank_1_config.device)

    save_obj(cpcfgan_trainer.G.state_dict(), rank_1_config.exp_dir + 'G.pt')
    save_obj(cpcfgan_trainer.D.state_dict(), rank_1_config.exp_dir + 'D.pt')

    """
    ----------------------- High Rank PCFGAN training -----------------------
    """
    config_dir = pt.join("configs/highrankCPCFGAN/{}.yaml".format(dataset))
    with open(config_dir) as file:
        rank_2_config = ml_collections.ConfigDict(yaml.safe_load(file))

    rank_2_config.exp_dir = './numerical_results/{}/grid_algo_{}_comment_{}/'.format(
        rank_2_config.dataset,
        rank_2_config.gan_algo,
        rank_2_config.comment)

    torch.manual_seed(0)
    np.random.seed(0)

    generator = ConditionalLSTMGenerator(input_dim=rank_1_config.data_feat_dim,
                                         output_dim=rank_1_config.data_feat_dim,
                                         hidden_dim=rank_1_config.G_hidden_dim,
                                         latent_dim=rank_1_config.G_latent_dim,
                                         noise_scale=rank_2_config.noise_scale,
                                         n_layers=2).to(rank_1_config.device)
    generator.load_state_dict(torch.load(rank_1_config.exp_dir + '/G.pt'))

    generator.eval()

    rank_1_pcf = pcf(num_samples=rank_1_config.Rank_1_num_samples,
                     hidden_size=rank_1_config.Rank_1_lie_degree,
                     input_dim=train_X.shape[-1],
                     add_time=True,
                     include_initial=False).to(rank_1_config.device)

    rank_1_pcf.load_state_dict(torch.load(rank_1_config.exp_dir + '/D.pt'))

    rank_1_pcf.eval()

    toggle_grad(generator, False)
    toggle_grad(rank_1_pcf, False)

    # Prepare real data
    test_size = int(0.2 * train_X.shape[0])
    test_data = train_X[-test_size:]
    train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl = prepare_dl_for_high_rank_pcfgan(rank_2_config, rank_1_pcf,
                                                                                                   train_X, test_data)
    batch_X, batch_X_dev_future = next(iter(train_reg_X_dl))
    print(batch_X.shape, batch_X_dev_future.shape)
    batch_X, past_dev_X = next(iter(train_pcf_X_dl))
    print(batch_X.shape, past_dev_X.shape)

    os.makedirs(rank_2_config.exp_dir, exist_ok=True)
    save_obj(rank_2_config, rank_2_config.exp_dir + '/config.pkl')
    save_obj(rank_1_config, rank_2_config.exp_dir + '/rank_1_config.pkl')
    torch.manual_seed(0)
    np.random.seed(0)

    high_rank_pcf_trainer = HighRankPCFGANTrainer(G=generator, rank_1_pcf=rank_1_pcf, config=rank_2_config)
    high_rank_pcf_trainer.reset_and_fit_regressors(train_reg_X_dl, test_reg_X_dl)
    if high_rank_pcf_trainer.tune_regression:
        high_rank_pcf_trainer.x_real_dl_for_regression_training(train_X)
    save_obj(high_rank_pcf_trainer.G.state_dict(), rank_2_config.exp_dir + 'original_G.pt')
    save_obj(high_rank_pcf_trainer.rank_1_pcf.state_dict(), rank_2_config.exp_dir + 'original_rank_1_pcf.pt')

    high_rank_pcf_trainer.fit(train_pcf_X_dl, rank_2_config.device)

    save_obj(high_rank_pcf_trainer.G.state_dict(), rank_2_config.exp_dir + 'G.pt')
    save_obj(high_rank_pcf_trainer.D.state_dict(), rank_2_config.exp_dir + 'D.pt')

    """
    ----------------------- Evaluation -----------------------
    """
    config_dir = pt.join("configs/evaluation_config_{}.yaml".format(dataset))
    with open(config_dir) as file:
        eval_config = ml_collections.ConfigDict(yaml.safe_load(file))
    eval_config.data_dir = './data/{}/'.format(dataset)

    test_X = load_obj(eval_config.data_dir + 'test_X.pt').float()
    test_X = test_X.to(eval_config.device)

    eval_config.n_lags = test_X.shape[1]
    eval_config.data_feat_dim = test_X.shape[-1]
    eval_config.past_path_length = 5
    eval_config.future_path_length = eval_config.n_lags - eval_config.past_path_length
    eval_config.result_name = '/results.csv'
    eval_config.exp_dir = rank_2_config.exp_dir
    torch.manual_seed(0)

    generator = ConditionalLSTMGenerator(input_dim=rank_2_config.data_feat_dim,
                                         output_dim=rank_2_config.data_feat_dim,
                                         hidden_dim=rank_2_config.G_hidden_dim,
                                         latent_dim=rank_2_config.G_latent_dim,
                                         noise_scale=rank_2_config.noise_scale,
                                         n_layers=2).to(rank_2_config.device)
    generator.load_state_dict(torch.load(rank_2_config.exp_dir + '/G.pt'))
    generator.eval()
    res_df = full_evaluation_latest(generator, test_X, eval_config)

    print(res_df)

    res_df.to_csv(eval_config.exp_dir + eval_config.result_name, index=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="fBM",
        help="choose from fBM, Stock",
    )

    parser.add_argument(
        "--gan_algo",
        type=str,
        default="CPCFGAN",
        help="choose from CPCFGAN,highrankCPCFGAN",
    )
    args = parser.parse_args()

    main(args.dataset)