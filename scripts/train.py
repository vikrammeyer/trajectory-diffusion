import argparse
from types import SimpleNamespace
from diff_traj.trainer import Trainer1D
from diff_traj.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D
from diff_traj.cfg import cfg
from knockknock import email_sender

# @email_sender(recipient_emails=["vjmeyer20@gmail.com", "vikram.j.meyer@vanderbilt.edu"], sender_email="trainstatus88@gmail.com")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--train_steps', type=int, default=700000)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    parser.add_argument('-st', '--sampling_timesteps', type=int, default=250)
    parser.add_argument('-l', '--loss_type', type=str, default='l2')
    parser.add_argument('-b', '--beta_schedule', type=str, default='linear')
    parser.add_argument('-d', '--dataset', type=str, default='/Users/vikram/research/tto/data/s2022/batch.csv') # small_data.csv
    parser.add_argument('-r', '--results', type=str, default='./results/test-train/')
    args = parser.parse_args()

    model = Unet1D(
        dim = cfg.traj_length,
        cond_dim = cfg.params_length,
        channels = 1,
        dim_mults = (1, 2, 4)
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = cfg.traj_length,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = args.loss_type,
        beta_schedule = args.beta_schedule
    )

    trainer = Trainer1D(
        diffusion,
        args.dataset,
        cfg,
        results_folder = args.results,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,              # gradient accumulation steps
        ema_decay = 0.995,                          # exponential moving average decay
    )

    trainer.train()

if __name__ == '__main__':
    main()
