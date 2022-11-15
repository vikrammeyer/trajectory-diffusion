import torch
import math
import torch.nn as nn
from accelerate import Accelerator
from diff_traj.dataset import StateDataset, ControlsDataset
from diff_traj.viz import Visualizations
from types import SimpleNamespace
from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int)
parser.add_argument('-r', '--results', type=str)
args = parser.parse_args()

cfg = SimpleNamespace(
    lane_width = 20,
    car_length = 5,
    car_width = 2,
    car_horizon = 60,
    dist_b4_obst = 15,
    min_obst_radius = 0,
    max_obst_radius = 0,
    n_obstacles = 3,
    n_constraints = 480,
    n_intervals = 40,
    interval_dur = 0.01,
    max_vel = 200,
    max_accel = 500,
    rng_seed = 0
)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FF(nn.Module):
    def __init__(self, in_dim, t_embed_size=32, param_size=13) -> None:
        super().__init__()

        self.c_size = t_embed_size + param_size

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim + self.c_size, in_dim // 2))

        self.layers.append(nn.Linear(in_dim // 2  + self.c_size, in_dim // 4))

        self.layers.append( nn.Linear(in_dim // 4 + self.c_size, in_dim // 4))

        self.layers.append(nn.Linear(in_dim // 4 + self.c_size, in_dim // 2))

        self.layers.append( nn.Linear(in_dim // 2 + self.c_size, in_dim))

        self.pos_emb = SinusoidalPosEmb(t_embed_size)

    def forward(self, x, t, params):
        c = torch.concat((self.pos_emb(t), params), dim=1)

        for layer in self.layers:
            x = layer(torch.concat((x, c), dim=1))

        return x


def noise_schedule(beta_start, beta_end, noise_steps, type):
    assert type in {'linear'}, f'noise schedule type {type} unsupported'

    if type == 'linear':
        return torch.linspace(beta_start, beta_end, noise_steps)

def sample_timesteps(n, noise_steps):
    return torch.randint(low=1, high=noise_steps, size=(n,))

def noise_data(alpha_hat, x, t):
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
    noise = torch.randn_like(x)

    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

results_dir = Path('../results')/args.results

if results_dir.exists():
    print('removed old results folder')
    shutil.rmtree(results_dir)

epochs = args.epochs
batch_size = 32
lr = 3e-4
noise_steps = 1000
beta_start = 1e-4
beta_end = 0.02

beta = noise_schedule(beta_start, beta_end, noise_steps, 'linear')
alpha = 1- beta
alpha_hat = torch.cumprod(alpha, dim=0)

accelerator = Accelerator()

model = FF(160)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
data = StateDataset('../../tto/data/s2022/batch.csv') # small_data.csv
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

mse = torch.nn.MSELoss()

for epoch in range(epochs):

    for batch_idx, (trajs, params) in enumerate(dataloader):
        optimizer.zero_grad()

        t = sample_timesteps(trajs.shape[0], noise_steps)

        x_t, noise = noise_data(alpha_hat, trajs, t)

        predicted_noise = model(x_t, t, params)

        loss = mse(noise, predicted_noise)
        accelerator.backward(loss)

        optimizer.step()

    print(f'finished epoch {epoch}')

accelerator.save_state(results_dir)

@torch.inference_mode()
def sample(param, model, traj_shape, noise_steps, alpha, alpha_hat, beta):
    x = torch.randn(traj_shape)

    for t in reversed(range(noise_steps)):
        t_vec = torch.ones((traj_shape[0])) * t+1
        predicted_noise = model(x, t_vec, param)

        if t == 0:
            noise = torch.zeros(traj_shape)
        else:
            noise = torch.randn(traj_shape)

        x = 1 / torch.sqrt(alpha[t]) * (x - ((1 - alpha[t]) / (torch.sqrt(1 - alpha_hat[t]))) * predicted_noise) + torch.sqrt(beta[t]) * noise

    return x

print(f'x: [{data.min_x}, {data.max_x}')
print(f'y: [{data.min_y}, {data.max_y}')
print(f'v: [{data.min_v}, {data.max_v}')

for j, (params, trajs) in enumerate(dataloader):
    if j > 0: break
    sampled_trajs = sample(params, model, trajs.shape, noise_steps, alpha, alpha_hat, beta).numpy()

    viz = Visualizations(cfg)

    params = params.numpy()

    for i in range(params.shape[0]):
        if i > 0: break
        print(sampled_trajs[i])
        traj = data.un_normalize(sampled_trajs[i])
        print('----')
        print(traj)
        viz.save_trajectory(traj, params[i][:4], params[i][4:], results_dir/f'{i}.png')