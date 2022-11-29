from diff_traj.utils.io import read_file
import pathlib
import torch
import math
import numpy as np

def clamp(n, smallest=-1, largest=1):
    return max(smallest, min(n, largest))

class StateDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_folder):
        super(StateDataset).__init__()
        param = []
        traj = []
        dataset_folder = pathlib.Path(dataset_folder)
        # TODO: save cfg in a seperate pkl file and load straight from here (maybe)
        for pkl_data_file in dataset_folder.glob('*.pkl'):
            data = read_file(pkl_data_file)

            for sample in data:
                param.append(sample['obsts'])
                traj.append(sample['states'])

        n_trajs = len(traj)
        traj_len = cfg.traj_length
        param_len = cfg.params_length

        min_x = 0
        max_x = 120
        min_y = -cfg.lane_width / 2
        max_y = cfg.lane_width / 2
        min_v = -cfg.max_vel
        max_v = cfg.max_vel

        min_r = cfg.min_obst_radius
        max_r = cfg.max_obst_radius

        trajs = torch.zeros((n_trajs, traj_len))
        params = torch.zeros((n_trajs, param_len))

        # Normalize the trajectories and obstacles to be in range [-1, 1] for each of their features
        for r in range(n_trajs):
            for c in range(0, traj_len, 4):
                trajs[r][c] = (traj[r][c] - min_x) / (max_x - min_x)     # Normalize x to [-1, 1]
                trajs[r][c+1] = (traj[r][c+1] - min_y) / (max_y - min_y) # Normalize y to [-1, 1]
                trajs[r][c+2] = (traj[r][c+2] - min_v) / (max_v - min_v) # Normalize v to [-1, 1]
                trajs[r][c+3] = math.cos(traj[r][c+3]) # encode the heading theta (radians) as cos theta

            # normalize the obstacles position (x,y)
            for c in range(4, param_len, 3):
                params[r][c] = (param[r][c] - min_x) / (max_x - min_x)
                params[r][c+1] = (param[r][c+1] - min_y) / (max_y - min_y)
                params[r][c+2] = (param[r][c+2] - min_r) / (max_r - min_r)

        self.n_trajs = n_trajs
        self.traj_len = traj_len
        self.param_len = param_len
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_v = min_v
        self.max_v = max_v
        self.min_r = min_r
        self.max_r = max_r
        self.trajs = trajs
        self.params = params

    def un_normalize(self, traj, params):
        # traj/params are not batched

        new_traj = np.zeros(self.traj_len)

        for c in range(0, self.traj_len, 4):
            new_traj[c] = (traj[c] * (self.max_x - self.min_x)) + self.min_x
            new_traj[c+1] = (traj[c+1] * (self.max_y - self.min_x)) + self.min_y
            new_traj[c+2] = (traj[c+2] * (self.max_v - self.min_v)) + self.min_v
            new_traj[c+3] = math.acos(clamp(traj[c+3])) # acos domain is [-1, 1] and predictions are noisy

        new_param = np.zeros(self.param_len)
        for c in range(0, self.param_len, 3):
            new_param[c] = params[c] * (self.max_x - self.min_x) + self.min_x
            new_param[c+1] = params[c+1] * (self.max_y - self.min_y) + self.min_y
            new_param[c+2] = params[c+2] * (self.max_r - self.min_r) + self.min_r

        return new_traj, new_param

    def __getitem__(self, idx):
        # unsqueeze to add a single channel dimension to work with Unet1D
        return self.trajs[idx][None, :], self.params[idx]

    def __len__(self):
        return self.n_trajs
