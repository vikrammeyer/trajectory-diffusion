from diff_traj.utils.io import read_file
import pathlib
import torch
import numpy as np

def neg_one_to_one(x, xmin, xmax):
    """ [xmin, xmax] -> [-1, 1]"""
    return 2 * ((x - xmin) / (xmax - xmin)) - 1

def zero_to_one(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def unnormalize(x_norm, xmin, xmax):
    return x_norm * (xmax - xmin) + xmin

def unnormalize_neg_one_to_one(x_norm, xmin, xmax):
    """ [-1, 1] -> [xmin, xmax] """
    return (((x_norm + 1) * (xmax - xmin)) / 2) + xmin

class StateDataset(torch.utils.data.Dataset):
    """ 1D temporal convolutions
        kernel_size: 16 (we probably want the kernel to look at 4 states each containing 4 elements)
        stride: 4
    """

    def __init__(self, cfg, dataset_folder):
        super(StateDataset).__init__()
        param = []
        traj = []
        dataset_folder = pathlib.Path(dataset_folder)
        for pkl_data_file in dataset_folder.glob('*.pkl'):
            data = read_file(pkl_data_file)

            for sample in data:
                param.append(sample['obsts'])
                traj.append(sample['states'])

        self.n_trajs = len(traj)
        self.traj_len = cfg.traj_length
        self.param_len = cfg.params_length

        self.min_x = 0
        self.max_x = 120
        self.min_y = -cfg.lane_width / 2
        self.max_y = cfg.lane_width / 2
        self.min_v = -cfg.max_vel
        self.max_v = cfg.max_vel
        self.min_theta = cfg.min_theta
        self.max_theta = cfg.max_theta

        self.min_r = cfg.min_obst_radius
        self.max_r = cfg.max_obst_radius

        self.trajs = torch.zeros((self.n_trajs, self.traj_len))
        self.params = torch.zeros((self.n_trajs, self.param_len))

        # Normalize the trajectories and obstacles to be in range [-1, 1] for each of their features
        for r in range(self.n_trajs):
            for c in range(0, self.traj_len, 4):
                self.trajs[r][c] = zero_to_one(traj[r][c], self.min_x, self.max_x)
                self.trajs[r][c+1] = zero_to_one(traj[r][c+1], self.min_y, self.max_y)
                self.trajs[r][c+2] = zero_to_one(traj[r][c+2], self.min_v, self.max_v)
                self.trajs[r][c+3] = zero_to_one(traj[r][c+3], self.min_theta, self.max_theta)

            for c in range(0,self. param_len, 3):
                self.params[r][c] = zero_to_one(param[r][c], self.min_x, self.max_x)
                self.params[r][c+1] = zero_to_one(param[r][c+1], self.min_y, self.max_y)
                self.params[r][c+2] = zero_to_one(param[r][c+2], self.min_r, self.max_r)

    def un_normalize(self, traj, params):
        # traj/params are not batched
        new_traj = np.zeros(self.traj_len)

        for c in range(0, self.traj_len, 4):
            new_traj[c] = unnormalize(traj[c], self.min_x, self.max_x)
            new_traj[c+1] = unnormalize(traj[c+1], self.min_y, self.max_y)
            new_traj[c+2] = unnormalize(traj[c+2], self.min_v, self.max_v)
            new_traj[c+3] = unnormalize(traj[c+3], self.min_theta, self.max_theta)

        new_param = np.zeros(self.param_len)
        for c in range(0, self.param_len, 3):
            new_param[c] = unnormalize(params[c], self.min_x, self.max_x)
            new_param[c+1] = unnormalize(params[c], self.min_y, self.max_y)
            new_param[c+2] = unnormalize(params[c], self.min_r, self.max_r)

        return new_traj, new_param

    def __getitem__(self, idx):
        # unsqueeze to add a single channel dimension to work with Unet1D
        return self.trajs[idx][None, :], self.params[idx]

    def __len__(self):
        return self.n_trajs

class StateChannelsDataset(torch.utils.data.Dataset):
    """ 1D temporal convolutions
        kernel size: 4
        stride: 1
    """

    def __init__(self, cfg, dataset_folder):
        super(StateChannelsDataset).__init__()
        param = []
        traj = []
        dataset_folder = pathlib.Path(dataset_folder)

        for pkl_data_file in dataset_folder.glob('*.pkl'):
            data = read_file(pkl_data_file)

            for sample in data:
                param.append(sample['obsts'])
                traj.append(sample['states'])

        self.n_trajs = len(traj)
        self.traj_len = cfg.traj_length
        self.param_len = cfg.params_length

        self.min_x = 0
        self.max_x = 120
        self.min_y = -cfg.lane_width / 2
        self.max_y = cfg.lane_width / 2
        self.min_v = -cfg.max_vel
        self.max_v = cfg.max_vel
        self.min_theta = cfg.min_theta
        self.max_theta = cfg.max_theta

        self.min_r = cfg.min_obst_radius
        self.max_r = cfg.max_obst_radius

        self.trajs = torch.zeros((self.n_trajs, self.traj_len))
        self.params = torch.zeros((self.n_trajs, self.param_len))

        # store each (x,y,v theta) in a separate channel for trajectories
        self.trajs = torch.zeros((self.n_trajs, 4, cfg.n_intervals))
        self.params = torch.zeros((self.n_trajs, self.param_len))

        # Normalize the trajectories and obstacles to be in range [-1, 1] for each of their features
        for r in range(self.n_trajs):
            for i, c in enumerate(range(0, self.traj_len, 4)):
                self.trajs[r][0][i] = zero_to_one(traj[r][c], self.min_x, self.max_x)
                self.trajs[r][1][i] = zero_to_one(traj[r][c+1], self.min_y, self.max_y)
                self.trajs[r][2][i] = zero_to_one(traj[r][c+2], self.min_v, self.max_v)
                self.trajs[r][3][i] = zero_to_one(traj[r][c+3], self.min_theta, self.max_theta)

            # normalize the obstacles position (x,y)
            for c in range(0, self.param_len, 3):
                self.params[r][c] = zero_to_one(param[r][c], self.min_x, self.max_x)
                self.params[r][c+1] = zero_to_one(param[r][c+1], self.min_y, self.max_y)
                self.params[r][c+2] = zero_to_one(param[r][c+2], self.min_r, self.max_r)

    def un_normalize(self, traj, params):
        # traj: C, N
        # traj/params are not batched

        new_traj = np.zeros(self.traj_len)

        for i, c in enumerate(range(0, self.traj_len, 4)):
            new_traj[c] = unnormalize(traj[0][i], self.min_x, self.max_x)
            new_traj[c+1] = unnormalize(traj[1][i], self.min_y, self.max_y)
            new_traj[c+2] = unnormalize(traj[2][i], self.min_v, self.max_v)
            new_traj[c+3] = unnormalize(traj[3][i], self.min_theta, self.max_theta)

        new_param = np.zeros(self.param_len)
        for c in range(0, self.param_len, 3):
            new_param[c] = unnormalize(params[c], self.min_x, self.max_x)
            new_param[c+1] = unnormalize(params[c], self.min_y, self.max_y)
            new_param[c+2] = unnormalize(params[c], self.min_r, self.max_r)

        return new_traj, new_param

    def __getitem__(self, idx):
        return self.trajs[idx], self.params[idx]

    def __len__(self):
        return self.n_trajs
