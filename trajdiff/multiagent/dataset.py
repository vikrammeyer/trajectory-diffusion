import torch
from trajdiff.utils import read_file
import glob
from torchvision.transforms import Compose, Lambda
from typing import Tuple
import numpy as np
import logging

def np_to_tensor(ndarray: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ndarray)

def list_to_tensor(lst: list) -> torch.Tensor:
    return torch.tensor(lst)

class MultiAgentDataset(torch.utils.data.Dataset):
    """
    trajectories of the shape: [40, 100, 2] (n_agents, n_steps, state_dim)
    dim0 -> agent index
    dim1 -> step in the trajectory
    dim2 -> element of the state vector (x is 1st, y is 2nd)
    """
    def __init__(self, data_folder, cfg, percent_history_of_full = 0.25):
        self.data = []

        for file in glob.glob(f'{data_folder}/*.pkl'):
            fdata = read_file(file)
            self.data.extend(fdata)

        self.n_agents = len(self.data[0]['trajectories'])
        self.traj_steps = len(self.data[0]['trajectories'][0])

        xmin = torch.ones(self.traj_steps) * cfg.xmin
        xmax = torch.ones(self.traj_steps) * cfg.xmax

        ymin = torch.ones(self.traj_steps) * cfg.ymin
        ymax = torch.ones(self.traj_steps) * cfg.ymax

        self.mintraj = torch.stack([xmin, ymin]).T.repeat(self.n_agents, 1, 1)
        self.maxtraj = torch.stack([xmax, ymax]).T.repeat(self.n_agents, 1, 1)

        self.percent_history_of_full = percent_history_of_full

        self.transforms = Compose([
            Lambda(list_to_tensor),
            Lambda(self.normalize_0_to_1),
            Lambda(self.split_history_and_future)
        ])

        logging.info('multiagent dataset loaded')

    def normalize_0_to_1(self, trajectories):
        return (trajectories - self.mintraj) / (self.maxtraj - self.mintraj)

    def unnormalize_0_to_1(self, trajectories):
        return trajectories * (self.maxtraj - self.mintraj) + self.mintraj

    def split_history_and_future(self, trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_len = trajectories.shape[1]
        history_steps = int(self.percent_history_of_full * traj_len)
        # split along the dim=1 (steps in the trajectory)
        return torch.split(trajectories, [history_steps, traj_len - history_steps], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ returns tuple of (history trajectory, future trajectory)
        """
        sample = self.data[idx]

        trajectories = sample['trajectories'] # list

        return self.transforms(trajectories)

if __name__ == '__main__':
    from trajdiff.multiagent import cfg
    ds = MultiAgentDataset('data/multiagent/test', cfg)
    dl = iter(ds)
    first, sec = next(dl)
    print(first.shape, sec.shape)