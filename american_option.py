import torch
import numpy as np
from src.model.generator.generator import ConditionalLSTMGenerator
import ml_collections
from os import path as pt
import yaml
from torch import nn
from src.utils import load_obj, american_put_pricer, LSM, to_numpy
import os
from src.baselines.TimeGAN import TimeGAN_module
from src.evaluations.summary import full_evaluation_latest
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(0)
device = 'cuda'

def main():
    config_dir = pt.join("configs/evaluation_config_Stock.yaml")
    with open(config_dir) as file:
        eval_config = ml_collections.ConfigDict(yaml.safe_load(file))
    eval_config.data_dir = './data/Stock/'

    train_X = load_obj(eval_config.data_dir+'train_X.pt').float()
    test_X = train_X.to(eval_config.device)
    test_X = load_obj(eval_config.data_dir+'test_X.pt').float()
    test_X = test_X.to(eval_config.device)
    eval_config.n_lags = test_X.shape[1]
    eval_config.data_feat_dim = test_X.shape[-1]
    eval_config.past_path_length = 5
    eval_config.future_path_length = eval_config.n_lags - eval_config.past_path_length

    dataset = 'Stock'
    models = ['RCGAN', 'TimeGAN', 'CPCFGAN', 'highrankCPCFGAN']
    for model in models:
        model_dir = 'pretrained_models/{}/{}'.format(dataset,model)
        exp_config = load_obj(model_dir + '/config.pkl')
        generator = ConditionalLSTMGenerator(input_dim=exp_config.data_feat_dim,
                                             output_dim=exp_config.data_feat_dim,
                                             hidden_dim=exp_config.G_hidden_dim,
                                             latent_dim=exp_config.G_latent_dim,
                                             noise_scale=exp_config.noise_scale,
                                             n_layers=2).to(exp_config.device)

        generator.load_state_dict(torch.load(model_dir + '/G.pt'))
        generator.eval()
        if model == 'TimeGAN':
            recovery = TimeGAN_module(
                input_dim=exp_config.input_dim,
                hidden_dim=exp_config.D_hidden_dim,
                out_dim=exp_config.input_dim,
                n_layers=exp_config.D_num_layers).to(exp_config.device)

            recovery.load_state_dict(torch.load(model_dir + '/recovery_state_dict.pt'))

            recovery.eval()

            supervisor = TimeGAN_module(
                input_dim=exp_config.input_dim,
                hidden_dim=exp_config.D_hidden_dim,
                out_dim=exp_config.input_dim,
                n_layers=exp_config.D_num_layers,
                activation=nn.Sigmoid()).to(exp_config.device)

            supervisor.load_state_dict(torch.load(model_dir + '/supervisor_state_dict.pt'))

            supervisor.eval()
        else:
            recovery = None
            supervisor = None
        sum_l1 = 0
        sum_var = 0
        for stock in range(5):
            price_real = LSM(increments = to_numpy(test_X[:,eval_config.past_path_length:]), idx=stock)
            prices_fake = []
            for idx in range(test_X.shape[0]):
                sample_real_X = test_X[idx, :, :]
                price_fake = american_put_pricer(generator=generator, past_path=sample_real_X, paths=20000,
                                                 N=eval_config.past_path_length, idx=stock,
                                                 supervisor=supervisor, recovery=recovery)
                prices_fake.append(price_fake)
            prices_fake = np.array(prices_fake)
            print(model, ' mean: ', np.abs(prices_fake.mean()))
            sum_l1 +=np.abs(price_real-prices_fake.mean())
            sum_var += prices_fake.std()**2

        print(model, ' mean: ', sum_l1/5, ' std: ', np.sqrt(sum_var)/5)


if __name__ == "__main__":
    main()