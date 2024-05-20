import torch
import numpy as np
from src.model.generator.generator import ConditionalLSTMGenerator
from src.model.discriminator.discriminators import LSTMDiscriminator
from torch.utils.data import DataLoader, Dataset
from src.baselines.RCGAN import RCGANTrainer
from src.baselines.TimeGAN import TIMEGANTrainer
from src.evaluations.summary import full_evaluation_latest
import ml_collections
import seaborn as sns
from os import path as pt
import yaml
from src.utils import load_obj, save_obj
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sns.set()
torch.manual_seed(0)
device = 'cuda'

TRAINERS = {"RCGAN": RCGANTrainer, "TimeGAN": TIMEGANTrainer}

def main(config):
    config.data_dir = 'data/{}/'.format(config.dataset)
    train_X = load_obj(config.data_dir+'train_X.pt').float().to(config.device)
    test_X = load_obj(config.data_dir+'test_X.pt').float().to(config.device)

    config.n_lags = train_X.shape[1]
    config.data_feat_dim = train_X.shape[-1]
    # For compatibility
    config.input_dim = config.data_feat_dim
    config.future_path_length = config.n_lags - config.past_path_length
    config.exp_dir = './numerical_results/{}/grid_algo_{}_comment_{}/'.format(
        config.dataset,
        config.gan_algo,
        config.comment)

    os.makedirs(config.exp_dir, exist_ok=True)
    save_obj(config, config.exp_dir + '/config.pkl')
    torch.manual_seed(0)
    np.random.seed(0)

    if config.gan_algo == "RCGAN":
        D_out_dim = 1
    else:
        D_out_dim = config.D_out_dim
    return_seq = True
    train_X_dl = DataLoader(train_X, config.batch_size, shuffle=True)

    generator = ConditionalLSTMGenerator(input_dim=config.data_feat_dim,
                                         output_dim=config.data_feat_dim,
                                         hidden_dim=config.G_hidden_dim,
                                         latent_dim=config.G_latent_dim,
                                         noise_scale=config.noise_scale,
                                         n_layers=2)

    if config.gan_algo == "RCGAN":
        discriminator = LSTMDiscriminator(
            input_dim=config.data_feat_dim,
            hidden_dim=config.D_hidden_dim,
            out_dim=D_out_dim,
            n_layers=config.D_num_layers,
            return_seq=return_seq,
        )
        trainer = RCGANTrainer(G=generator,
                               D=discriminator,
                               train_dl=train_X_dl,
                               batch_size=config.batch_size,
                               n_gradient_steps=config.steps,
                               config=config, )
    elif config.gan_algo == "TimeGAN":
        trainer = TIMEGANTrainer(
            G=generator,
            gamma=1,
            train_dl=train_X_dl,
            batch_size=config.batch_size,
            n_gradient_steps=config.steps,
            config=config,
        )
    else:
        raise ValueError("Unknown algorithm")

    save_obj(config, pt.join(config.exp_dir, "config.pkl"))
    trainer.fit(config.device)

    save_obj(trainer.G.state_dict(), config.exp_dir + 'G.pt')

    if config.gan_algo == "TimeGAN":
        save_obj(
            trainer.recovery.state_dict(),
            pt.join(config.exp_dir, "recovery_state_dict.pt"),
        )
        save_obj(
            trainer.supervisor.state_dict(),
            pt.join(config.exp_dir, "supervisor_state_dict.pt"),
        )
        save_obj(
            trainer.embedder.state_dict(),
            pt.join(config.exp_dir, "embedder_state_dict.pt"),
        )

    """
    Evaluation
    """
    config_dir = pt.join("configs/evaluation_config_{}.yaml".format(config.dataset))
    with open(config_dir) as file:
        eval_config = ml_collections.ConfigDict(yaml.safe_load(file))
    eval_config.data_dir = './data/{}/'.format(config.dataset)

    test_X = load_obj(eval_config.data_dir + 'test_X.pt').float()
    test_X = test_X.to(eval_config.device)

    eval_config.n_lags = test_X.shape[1]
    eval_config.data_feat_dim = test_X.shape[-1]
    eval_config.past_path_length = 5
    eval_config.future_path_length = eval_config.n_lags - eval_config.past_path_length
    eval_config.result_name = '/results.csv'
    eval_config.exp_dir = config.exp_dir
    torch.manual_seed(0)
    trainer.G.eval()
    if config.gan_algo == "TimeGAN":
        trainer.recovery.eval()
        trainer.supervisor.eval()
        res_df = full_evaluation_latest(trainer.G, test_X, eval_config, recovery=trainer.recovery, supervisor=trainer.supervisor)
    else:
        res_df = full_evaluation_latest(trainer.G, test_X, eval_config)

    print(res_df)

    res_df.to_csv(eval_config.exp_dir + eval_config.result_name, index=True)

if __name__ == "__main__":
    datasets = ['fBM', 'Stock']
    models = ['RCGAN', 'TimeGAN']
    for dataset in datasets:
        for algo in models:
            config_dir = pt.join("configs/{}/{}.yaml".format(algo, dataset))
            with open(config_dir) as file:
                config = ml_collections.ConfigDict(yaml.safe_load(file))
            config.device = device
            config.dataset = dataset
            config.gan_algo = algo
            main(config)