import argparse
import logging
from pathlib import Path

from trajdiff.diffusion import Trainer1D, Unet1D, GaussianDiffusion1D
from trajdiff.dataset import StateDataset
from trajdiff import cfg
from trajdiff.utils import setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_folder', default='results/train/')
    parser.add_argument('-o', '--output_folder', default='./results/trajdiff/')
    parser.add_argument('-n', '--train_steps', type=int, default=50000)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    parser.add_argument('-st', '--sampling_timesteps', type=int, default=25)
    parser.add_argument('-l', '--loss_type', default='l2', help='l1, l2')
    parser.add_argument('-b', '--beta_schedule', default='cosine', help='linear, cosine')
    parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok = True)
    setup_logging(args.log_level, True, output_folder/f"train-diffusion.log")

    channels = 1
    seq_length = cfg.traj_length
    dataset = StateDataset(cfg, args.dataset_folder)

    model = Unet1D(
        dim = 64,
        cond_dim = cfg.params_length,
        channels = channels,
        cond_drop_prob=0.05
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_length,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,   # using ddim for faster inference
        loss_type = args.loss_type,
        beta_schedule = args.beta_schedule
    )

    logging.info('built gaussian diffusion')

    trainer = Trainer1D(
        diffusion,
        dataset,
        results_folder = output_folder,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = args.train_steps,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
    )

    logging.info('built trainer')

    trainer.train()

    logging.info("finished training")

if __name__ == '__main__':
    main()
