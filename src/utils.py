import torch
import torch.nn as nn
import os
import pickle
import yaml
import ml_collections
import re
import numpy as np
from torch.nn.functional import one_hot

def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(
        dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def AddTime(x):
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


def get_time_vector_4d(size: int, channel: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, channel, 1, 1)


def AddTime_4d(x):
    t = get_time_vector_4d(x.shape[0], x.shape[1], x.shape[2]).to(x.device)
    return torch.cat([t, x], dim=-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass


def track_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def track_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def construct_future_dev_path(pcf, path):
    """
    Construct the future development path
    Input:
    pcf: Rank 1 pcf layer
    path: tensor of shape [N, T, d]
    Output:
    development path of shape [N, m, T, lie, lie]
    """
    with torch.no_grad():
        lie_degree = pcf.degree
        num_samples = pcf.num_samples
        N, T, D = path.shape
        steps = T
        dev_list = []
        for step in range(steps):
            if step == steps-1:
                dev = torch.eye(lie_degree).repeat(N, num_samples, 1, 1).to(dtype=dev_list[0].dtype, device=dev_list[0].device)
                dev_list.append(dev)
            else:
                dev_list.append(pcf.unitary_development(path[:, step:]))
    return torch.stack(dev_list).permute([1,2,0,3,4]).squeeze()

def construct_past_dev_path(pcf, path):
    """
    Construct the past development path
    Input:
    pcf: Rank 1 pcf layer
    path: tensor of shape [N, T, d]
    Output:
    development path of shape [N, m, T, lie, lie]
    """
    with torch.no_grad():
        lie_degree = pcf.degree
        num_samples = pcf.num_samples
        N, T, D = path.shape
        steps = T
        dev_list = []
        for step in range(1, steps+1):
            if step == 1:
                dev = torch.eye(lie_degree).repeat(N, num_samples, 1, 1)
                dev_list.append(dev)
            else:
                dev_list.append(pcf.unitary_development(path[:, :step]))
        dev_list[0] = dev_list[0].to(dtype=dev_list[-1].dtype, device=dev_list[-1].device)
    return torch.stack(dev_list).permute([1,2,0,3,4]).squeeze()

def get_experiment_dir(config):
    exp_dir = './numerical_results/{dataset}/algo_{gan}_G_{generator}_D_lie_degree_{liedeg}_n_lag_{n_lags}_{seed}_comment_{comment}_{lie_group}'.format(
        dataset=config.dataset, gan=config.gan_algo, generator=config.generator,
        liedeg=config.Rank_2_lie_degree, n_lags=config.n_lags, seed=config.seed, comment=config.comment, lie_group=config.lie_group)
    os.makedirs(exp_dir, exist_ok=True)
    if config.train and os.path.exists(exp_dir):
        print("WARNING! The model exists in directory and will be overwritten")
    config.exp_dir = exp_dir


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass

def load_config(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config

def track_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def track_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def set_seed(seed: int, device='cpu'):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cupy.random.seed(seed)

    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x)
    return torch.cat(tensor)

def loader_to_tensor_(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def loader_to_cond_tensor(dl, config):
    tensor = []
    for _, y in dl:
        tensor.append(y)

    return one_hot(torch.cat(tensor), config.num_classes).unsqueeze(1).repeat(1, config.n_lags, 1)

def combine_dls(dls):
    return torch.cat([loader_to_tensor_(dl) for dl in dls])


def is_multivariate(x: torch.Tensor):
    """ Check if the path / tensor is multivariate. """
    return True if x.shape[-1] > 1 else False


def find_folders_with_pattern(directory, pattern):
    """
    Search for folders within a given directory that match a specific pattern.
    """
    matching_folders = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if re.search(pattern, dir_name):
                matching_folders.append(os.path.join(root, dir_name))
    return matching_folders

def flatten_complex_matrix(dev):
    """
    Flatten complex matrix of shape NxN to 2N^2
    Parameters
    ----------
    dev

    Returns
    -------

    """
    batch_size, num_samples, T, lie_deg, _ = dev.shape
    dev = dev.reshape([batch_size, num_samples, T, lie_deg ** 2])
    flat_dev = torch.cat([dev.real, dev.imag], -1)
    return flat_dev


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_test_errors_empirical(H0_stats, H1_stats, alpha, test_type='one-sided'):
    """
    Compute the power, Type I error (α), and Type II error (β) of a hypothesis test
    using empirical quantiles from H0_stats.

    :param H0_stats: Test statistics under the null hypothesis as a numpy array
    :param H1_stats: Test statistics under the alternative hypothesis as a numpy array
    :param alpha: Significance level (Type I error)
    :param test_type: Type of the hypothesis test ('one-sided' or 'two-sided')

    :return: Dictionary with Type I error, Type II error, and Power of the test
    """
    if test_type == 'one-sided':
        # Critical value based on the upper quantile for one-sided test
        critical_value = np.quantile(H0_stats, 1 - alpha)
    else:
        # Critical values for two-sided test
        critical_value_lower = np.quantile(H0_stats, alpha / 2)
        critical_value_upper = np.quantile(H0_stats, 1 - alpha / 2)

    if test_type == 'one-sided':
        type_I_error = np.mean(H0_stats > critical_value)
        type_II_error = np.mean(H1_stats <= critical_value)
    else:
        type_I_error = np.mean((H0_stats < critical_value_lower) | (H0_stats > critical_value_upper))
        type_II_error = np.mean((H1_stats >= critical_value_lower) & (H1_stats <= critical_value_upper))

    power = 1 - type_II_error

    return power, type_I_error, type_II_error


def fbm_cov(H, timesteps):
    time_grid = torch.range(0,1,1/timesteps)
    cov_matrix = torch.zeros([timesteps+1, timesteps+1])
    for i in range(timesteps+1):
        for j in range(i, timesteps+1):
            cov_matrix[i,j] = 0.5*(time_grid[i]**(2*H)+time_grid[j]**(2*H)-(time_grid[j]-time_grid[i])**(2*H))
            cov_matrix[j,i] = cov_matrix[i,j]
    return cov_matrix

def compute_expected_mean(x_real, H, timesteps, config):
    device = config.device
    cov_matrix = fbm_cov(H, timesteps)[1:,1:]
    sigma_12 = cov_matrix[config.past_path_length-1:,:config.past_path_length-1].to(device)
    sigma_22 = cov_matrix[:config.past_path_length-1,:config.past_path_length-1].to(device)
    exp_future = sigma_12@torch.linalg.inv(sigma_22).to(device)@x_real[1:config.past_path_length]
    x_real_ = torch.cat([x_real[:config.past_path_length],exp_future])
    return x_real_

def american_put_pricer(generator, past_path, paths=10000, N = 5, idx = 0, supervisor=None, recovery=None):
    past_path_repeat = past_path.repeat((paths, 1, 1))
    if supervisor and recovery:
        increments = generator(N, past_path_repeat[:, 10 - N - 5:10 - N])
        increments = supervisor(increments)
        increments = to_numpy(recovery(increments))
    else:
        increments = to_numpy(generator(N, past_path_repeat[:, 10 - N - 5:10 - N]))

    return LSM(increments, idx)


def LSM(increments, idx=0):
    """
    Longstaff-Schwartz Method for pricing American options

    N = number of time steps
    paths = number of generated paths
    order = order of the polynomial for the regression
    """
    N, T, D = increments.shape
    S0 = 100  # Initial price
    K = 100  # Strike price
    # T = T / 252
    dt = 1 / 252  # time interval
    r = 0.01
    payoff = "put"
    df = np.exp(r * dt)  # discount factor per time interval
    X0 = np.zeros((N, 1))

    X = np.concatenate((X0, increments[:,:,idx]), axis=1).cumsum(1)
    S = S0 * np.exp(X)

    H = np.maximum(K - S, 0)  # intrinsic values for put option
    #     print(H.shape)
    V = np.zeros_like(H)  # value matrix
    V[:, -1] = H[:, -1]

    # Valuation by LS Method
    for t in range(T - 2, 0, -1):
        good_paths = H[:, t] > 0
        if not S[good_paths].any():
            continue
        rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, 2)  # polynomial regression
        C = np.polyval(rg, S[good_paths, t])  # evaluation of regression

        exercise = np.zeros(len(good_paths), dtype=bool)
        exercise[good_paths] = H[good_paths, t] > C

        V[exercise, t] = H[exercise, t]
        V[exercise, t + 1:] = 0
        discount_path = V[:, t] == 0
        V[discount_path, t] = V[discount_path, t + 1] * df

    V0 = np.mean(V[:, 1]) * df  #
    return V0