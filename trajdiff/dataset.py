import torch
from trajdiff.utils import read_file
import glob
from torchvision.transforms import Compose, Lambda
from typing import Tuple
import numpy as np

X_LIMS = (0, 800)
Y_LIMS = (0, 800)

def zero_to_one(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def unnormalize(x_norm, xmin, xmax):
    return x_norm * (xmax - xmin) + xmin

def np_to_tensor(ndarray: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ndarray)

def list_to_tensor(lst: list) -> torch.Tensor:
    return torch.tensor(lst)

# TODO: figure out how to make these dynamically sized to the trajectory length
# at the start of each run but not have to be recomputed in every normalize call
# -> likely integrate with the dataset class is the answer (also the XLIMS and YLIMS)


# min_t = torch.zeros(100)
# max_t = torch.zeros(100)

# for i in range(100):
#     if i % 2 == 0:
#         min_t[i] = X_LIMS[0]
#         max_t[i] = X_LIMS[1]
#     else:
#         min_t[i] = Y_LIMS[0]
#         max_t[i] = Y_LIMS[1]

# # need to replicate min/max traj for each agent in dim0 of trajectories
# def normalize_0_to_1(trajectories):
#     mint = min_t.repeat(trajectories.shape[0],1)
#     maxt = max_t.repeat(trajectories.shape[0],1)

#     return (trajectories - mint) / (maxt - mint)

# def unnormalize_0_to_1(trajectories):
#     mint = min_t.repeat(trajectories.shape[0],1)
#     maxt = max_t.repeat(trajectories.shape[0],1)

#     return trajectories * (maxt - mint) + mint

# def split_history_and_future(trajectories: torch.Tensor, percent_history_of_full = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
#     traj_len = trajectories.shape[1]
#     history_steps = int(percent_history_of_full * traj_len)
#     # split along the dim=1 (steps in the trajectory)
#     return torch.split(trajectories, [history_steps, traj_len - history_steps], dim=1)

# transforms = Compose([
#     Lambda(list_to_tensor),
#     Lambda(normalize_0_to_1),
#     Lambda(split_history_and_future)
# ])

class MultiAgentDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, percent_history_of_full = 0.25):
        self.data = []

        for file in glob.glob(f'{data_folder}/*.pkl'):
            fdata = read_file(file)
            self.data.extend(fdata)

        self.n_agents = len(self.data[0]['trajectories'])
        self.traj_steps = len(self.data[0]['trajectories'][0])

        self.min_t = torch.zeros(self.traj_steps)
        self.max_t = torch.zeros(self.traj_steps)

        # xmin = torch.ones(self.traj_steps) * X_LIMS[0]
        # xmax = torch.ones(self.traj_steps) * X_LIMS[1]

        # ymin = torch.ones(self.traj_steps)

        for i in range(self.traj_steps):
            if i % 2 == 0:
                self.min_t[i] = X_LIMS[0]
                self.max_t[i] = X_LIMS[1]
            else:
                self.min_t[i] = Y_LIMS[0]
                self.max_t[i] = Y_LIMS[1]

        self.percent_history_of_full = percent_history_of_full

        self.transforms = Compose([
            Lambda(list_to_tensor),
            Lambda(self.normalize_0_to_1),
            Lambda(self.split_history_and_future)
        ])

    # need to replicate min/max traj for each agent in dim0 of trajectories
    def normalize_0_to_1(self, trajectories):
        # BUG: need to think deeper about what to do here for how to optimally load in down the line
        # i think the way the trajectories is stored works well currently
        # 1 channel for x traj, 1 channel for y traj
        # normalize each channel pretty easily separately
        mint = self.min_t.repeat(trajectories.shape[0],1)
        maxt = self.max_t.repeat(trajectories.shape[0],1)
        print(trajectories.shape, mint.shape, maxt.shape)
        # torch.Size([40, 100, 2]) torch.Size([40, 100]) torch.Size([40, 100])
        return (trajectories - mint) / (maxt - mint)

    def unnormalize_0_to_1(self, trajectories):
        mint = self.min_t.repeat(trajectories.shape[0],1)
        maxt = self.max_t.repeat(trajectories.shape[0],1)

        return trajectories * (maxt - mint) + mint

    def split_history_and_future(self, trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_len = trajectories.shape[1]
        history_steps = int(self.percent_history_of_full * traj_len)
        # split along the dim=1 (steps in the trajectory)
        return torch.split(trajectories, [history_steps, traj_len - history_steps], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # for a list of 3 lists of size 4 -> (3,4) tensor
        # -> each nested list becomes a row in the tensor
        # dim0 (row) -> agent index
        # dim1 (col) -> step in the trajectory

        trajectories = sample['trajectories']
        #radii = torch.FloatTensor(sample['radii']) # (only dim is agent)

        return self.transforms(trajectories)

if __name__ == '__main__':
    a = [[i for i in range(100)] for _ in range(3)] # 3 agents, 100 steps in trajectory

    # print(transforms(a))

    ds = MultiAgentDataset('data/multiagent/test')
    dl = iter(ds)
    print(next(dl))
    # print(ds.transforms(a))