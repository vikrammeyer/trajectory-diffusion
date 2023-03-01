import logging
import torch
import argparse
from pathlib import Path

from trajdiff.utils import set_seed, setup_logging
from trajdiff.baselines import FCNet, train_baseline, save_results
from trajdiff.dataset import StateDataset
from trajdiff import cfg

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save_folder', required=True)
parser.add_argument('-td', '--train_data', required=True)
parser.add_argument('-vd', '--val_data', required=True)
parser.add_argument('-l', '--layers', nargs="+", type=int, help="Enter layer sizes like -s 1 2 3", default=[9,256,512,1024,240])
parser.add_argument('-e','--epochs', type=int, default=1000)
parser.add_argument('-b','--batchsize', type=int, default=64)
parser.add_argument('-lr','--learning_rate', type=float, default=3e-4)
parser.add_argument('-ct', '--convergence_threshold', type=float, default=1e-3)
parser.add_argument('--seed',type=int, default=42)
parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
args = parser.parse_args()

save_folder = Path(f'results/{args.save_folder}')
if not save_folder.exists():
    save_folder.mkdir()

setup_logging(args.log_level, True, save_folder/"baseline_training.log")

set_seed(args.seed)

model = FCNet(args.layers)

train_data = StateDataset(cfg, args.train_data)
val_data = StateDataset(cfg, args.val_data)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=True)

trained_model, losses = train_baseline(model, train_loader, val_loader, args.epochs, args.learning_rate)

save_results(trained_model, losses, save_folder, args)
