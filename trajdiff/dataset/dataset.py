import logging
import pathlib

import numpy as np
import torch

from trajdiff.utils import read_file


def zero_to_one(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def unnormalize(x_norm, xmin, xmax):
    return x_norm * (xmax - xmin) + xmin


class StateDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_folder, channel_dim=False):
        super(StateDataset).__init__()
        self.channel_dim = channel_dim

        param = []
        traj = []
        dataset_folder = pathlib.Path(dataset_folder)
        for pkl_data_file in dataset_folder.glob("*.pkl"):
            data = read_file(pkl_data_file)

            for sample in data:
                param.append(sample["obsts"])
                traj.append(sample["states"])

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
                self.trajs[r][c + 1] = zero_to_one(
                    traj[r][c + 1], self.min_y, self.max_y
                )
                self.trajs[r][c + 2] = zero_to_one(
                    traj[r][c + 2], self.min_v, self.max_v
                )
                self.trajs[r][c + 3] = zero_to_one(
                    traj[r][c + 3], self.min_theta, self.max_theta
                )

            for c in range(0, self.param_len, 3):
                self.params[r][c] = zero_to_one(param[r][c], self.min_x, self.max_x)
                self.params[r][c + 1] = zero_to_one(
                    param[r][c + 1], self.min_y, self.max_y
                )
                self.params[r][c + 2] = zero_to_one(
                    param[r][c + 2], self.min_r, self.max_r
                )

        logging.info("dataset loaded")

    def normalize(self, params):
        # not batched
        params = torch.tensor(params)
        normed_params = torch.zeros_like(params)
        for c in range(0, self.param_len, 3):
            normed_params[c] = zero_to_one(params[c], self.min_x, self.max_x)
            normed_params[c + 1] = zero_to_one(params[c + 1], self.min_y, self.max_y)
            normed_params[c + 2] = zero_to_one(params[c + 2], self.min_r, self.max_r)
        return normed_params

    def unnormalize(self, traj, params):
        # traj/params are not batched
        new_traj = np.zeros(self.traj_len)

        for c in range(0, self.traj_len, 4):
            new_traj[c] = unnormalize(traj[c], self.min_x, self.max_x)
            new_traj[c + 1] = unnormalize(traj[c + 1], self.min_y, self.max_y)
            new_traj[c + 2] = unnormalize(traj[c + 2], self.min_v, self.max_v)
            new_traj[c + 3] = unnormalize(traj[c + 3], self.min_theta, self.max_theta)

        new_param = np.zeros(self.param_len)
        for c in range(0, self.param_len, 3):
            new_param[c] = unnormalize(params[c], self.min_x, self.max_x)
            new_param[c + 1] = unnormalize(params[c + 1], self.min_y, self.max_y)
            new_param[c + 2] = unnormalize(params[c + 2], self.min_r, self.max_r)

        return new_traj, new_param

    def __getitem__(self, idx):
        if self.channel_dim:
            # unsqueeze to add a single channel dimension to work with Unet1D
            return self.params[idx], self.trajs[idx][None, :]

        return self.params[idx], self.trajs[idx]

    def __len__(self):
        return self.n_trajs
