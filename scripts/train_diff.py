import argparse
import logging
from pathlib import Path
from datetime import datetime
from diff_traj.diffusion.trainer import Trainer1D
from diff_traj.dataset.dataset import StateChannelsDataset, StateDataset
from diff_traj.diffusion.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D
from diff_traj.cfg import cfg
from diff_traj.utils.logs import setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--train_steps', type=int, default=50000)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    parser.add_argument('-st', '--sampling_timesteps', type=int, default=25)
    parser.add_argument('-l', '--loss_type', default='l2', help='l1, l2')
    parser.add_argument('-b', '--beta_schedule', default='cosine', help='linear, cosine')
    parser.add_argument('-dt', '--dataset_type', default='state', help='state, channels')
    parser.add_argument('-d', '--dataset_folder', default='./data/subset/')
    parser.add_argument('-o', '--output_folder', default='./results/demo/')
    parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok = True)
    now = datetime.now().strftime("%b-%d-%H-%M-%S")
    setup_logging(args.log_level, True, output_folder/f"train-diff-{now}.log")

    if args.dataset_type == 'state':
        channels = 1
        dataset = StateDataset(cfg, args.dataset_folder)
    elif args.dataset_type == 'channels':
        channels = 4
        dataset = StateChannelsDataset(cfg, args.dataset_folder)
    else:
        raise ValueError("dataset type not supported")

    logging.info('loaded dataset')

    model = Unet1D(
        dim = 64,
        cond_dim = cfg.params_length,
        channels = channels,
    )

    logging.info('built unet')

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = cfg.traj_length,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = args.loss_type,
        beta_schedule = args.beta_schedule
    )

    logging.info('built gaussian diffusion')

    trainer = Trainer1D(
        diffusion,
        dataset,
        cfg,
        results_folder = output_folder,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,              # gradient accumulation steps
        ema_decay = 0.995,                          # exponential moving average decay
    )

    logging.info('built trainer')

    trainer.train()

    logging.info("finished training")

if __name__ == '__main__':
    main()
