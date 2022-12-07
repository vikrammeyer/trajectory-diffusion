from diff_traj.dataset.dataset import StateDataset
from diff_traj.diffusion.classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D
from diff_traj.diffusion.trainer import Trainer1D
from diff_traj.utils.io import write_obj
from diff_traj.cfg import cfg
import logging
from datetime import datetime
from diff_traj.utils.logs import setup_logging
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--milestone_folder',required=True)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    parser.add_argument('-st', '--sampling_timesteps', type=int, default=25)
    parser.add_argument('-l', '--loss_type', default='l2', help='l1, l2')
    parser.add_argument('-b', '--beta_schedule', default='cosine', help='linear, cosine')
    parser.add_argument('-d', '--dataset_folder', default='./data/subset/')
    parser.add_argument('-o', '--output_folder', default='./results/demo/')
    parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok = True)
    now = datetime.now().strftime("%b-%d-%H-%M-%S")
    setup_logging(args.log_level, True, output_folder/f"sample-diff-{now}.log")

    channels=1
    dataset = StateDataset(cfg, args.dataset_folder)

    logging.info('loaded dataset')

    model = Unet1D(
        dim = 64,
        cond_dim = cfg.params_length,
        channels = channels,
        cond_drop_prob=0.05
    )

    logging.info('built unet')

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = cfg.traj_length,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,
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
        train_lr = 0,
        train_num_steps = 0,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        debug_mode = True
    )

    for checkpoint in range(1, 100):
        trainer.load(f'{args.milestone_folder}/model-{checkpoint}.pt')
        for i, obj in enumerate(trainer.sample(dataset), 1):
            write_obj(obj, output_folder/f'sampled-{i}.pkl')

        logging.info('finished logging checkpoint %s', checkpoint)

if __name__ == '__main__':
    main()
