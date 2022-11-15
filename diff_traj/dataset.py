import torch
import math
import csv
import numpy as np

def clamp(n, smallest=-1, largest=1):
    return max(smallest, min(n, largest))

class ControlsDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        #TODO: implement std normalization x - min / max for accel and ang velocity
        super(ControlsDataset).__init__()

        self.x = []
        self.y = []

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.x.append([float(i) for i in row[:13]])
                self.y.append([float(i) for i in row[13+160:13+160+80]])

        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

class StateDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(StateDataset).__init__()

        param = []
        traj = []

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                param.append([float(i) for i in row[:13]])
                traj.append([float(i) for i in row[13:13+160]])

        n_trajs = len(traj)
        traj_len = len(traj[0])
        param_len = len(param[0])

        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf
        min_v = math.inf
        max_v = -math.inf
        for r in range(n_trajs):
            for c in range(0, traj_len, 4):
                min_x = min(min_x, traj[r][c])
                max_x = max(max_x, traj[r][c])
                min_y = min(min_y, traj[r][c+1])
                max_y = max(max_y, traj[r][c+1])
                min_v = min(min_v, traj[r][c+2])
                max_v = max(max_v, traj[r][c+2])

        trajs = torch.zeros((n_trajs, traj_len))
        params = torch.zeros((n_trajs, param_len))
        for r in range(n_trajs):
            for c in range(0, traj_len, 4):
                trajs[r][c] = (traj[r][c] - min_x) / (max_x - min_x)   # Normalize x to [-1, 1]
                trajs[r][c+1] = (traj[r][c+1] - min_y) / (max_y - min_y) # Normalize y to [-1, 1]
                trajs[r][c+2] = (traj[r][c+2] - min_v) / (max_v - min_v) # Normalize v to [-1, 1]
                trajs[r][c+3] = math.cos(traj[r][c+3]) # encode the heading theta (radians) as cos theta

            for c in range(4, param_len, 3):
                params[r][c+1] = (param[r][c] - min_x) / (max_x - min_x)
                params[r][c+2] = (param[r][c] - min_y) / (max_y - min_y)

        self.n_trajs = n_trajs
        self.traj_len = traj_len
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_v = min_v
        self.max_v = max_v
        self.trajs = trajs
        self.params = params

    def un_normalize(self, traj) -> np.ndarray:
        # traj is not a batch

        new_traj = np.zeros(self.traj_len)

        for c in range(0, self.traj_len, 4):
            new_traj[c] = (traj[c] * (self.max_x - self.min_x)) + self.min_x
            new_traj[c+1] = (traj[c+1] * (self.max_y - self.min_x)) + self.min_y
            new_traj[c+2] = (traj[c+2] * (self.max_v - self.min_v)) + self.min_v
            new_traj[c+3] = math.acos(clamp(traj[c+3])) # acos domain is [-1, 1] and predictions are noisy
            # print(f'cos(theta) = {traj[c+3]}')

        return new_traj

    def __getitem__(self, idx):
        # unsqueeze to add a single channel dimension
        return self.trajs[idx][None, :], self.params[idx]

    def __len__(self):
        return self.n_trajs