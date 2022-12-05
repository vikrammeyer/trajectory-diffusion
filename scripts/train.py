import argparse
from diff_traj.trainer import Trainer1D
from diff_traj.dataset.dataset import StateChannelsDataset, StateDataset
from diff_traj.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D
from diff_traj.cfg import cfg
from knockknock import email_sender

# @email_sender(recipient_emails=["vjmeyer20@gmail.com", "vikram.j.meyer@vanderbilt.edu"], sender_email="trainstatus88@gmail.com")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--train_steps', type=int, default=50000)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    parser.add_argument('-st', '--sampling_timesteps', type=int, default=25)
    parser.add_argument('-l', '--loss_type', type=str, default='l2')
    parser.add_argument('-b', '--beta_schedule', type=str, default='cosine')
    parser.add_argument('-dt', '--dataset_type', type=str, default='state')
    parser.add_argument('-d', '--dataset_path', type=str, default='/Users/vikram/research/tto/data/s2022/batch.csv') # small_data.csv
    parser.add_argument('-r', '--results', type=str, default='./results/test-train/')
    args = parser.parse_args()

    if args.dataset_type == 'state':
        init_kernel = 16
        init_stride = 4
        channels = 1
        dataset = StateDataset(cfg, args.dataset_path)
    elif args.dataset_type == 'channels':
        init_kernel = 4
        init_stride = 1
        channels = 4
        dataset = StateChannelsDataset(cfg, args.dataset_path)
    else:
        raise ValueError("dataset type not supported")

    model = Unet1D(
        dim = cfg.traj_length,
        cond_dim = cfg.params_length,
        init_kernel = init_kernel,
        init_stride = init_stride,
        channels = channels,
        dim_mults = (1, 2, 4),
        resnet_block_groups=4
    )

    print('built unet')

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = cfg.traj_length,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = args.loss_type,
        beta_schedule = args.beta_schedule
    )

    print('build gaussian diffusion')

    trainer = Trainer1D(
        diffusion,
        dataset,
        cfg,
        results_folder = args.results,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,              # gradient accumulation steps
        ema_decay = 0.995,                          # exponential moving average decay
    )

    print('built trainer')

    trainer.train()

if __name__ == '__main__':
    main()
