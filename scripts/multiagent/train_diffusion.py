import argparse
import logging
import traceback

from trajdiff.diffusion import Unet1D, GaussianDiffusion1D, train
from trajdiff.multiagent import MultiAgentDataset, cfg
from trajdiff.utils import setup
from trajdiff.diffusion.set_transformer import SetTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_folder', required=True)
    parser.add_argument('-o', '--output_folder', required=True)
    parser.add_argument('-p', '--percent_history', type=float, default=0.24, help="x\% of trajectory to use as the history")
    parser.add_argument('-c', '--cond_dim', type=int, default=512)
    parser.add_argument('-n', '--train_steps', type=int, default=50000)
    parser.add_argument('-t', '--timesteps', type=int, default=1000)
    parser.add_argument('-st', '--sampling_timesteps', type=int, default=25)
    parser.add_argument('-l', '--loss_type', default='l2', help='l1, l2')
    parser.add_argument('-b', '--beta_schedule', default='cosine', help='linear, cosine')
    parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
    args = parser.parse_args()

    output_folder = setup(args, "train-diffusion.log")

    dataset = MultiAgentDataset(args.dataset_folder, cfg, args.percent_history)
    channels = 2
    seq_length = dataset.traj_steps

    history_seq_len = int(args.percent_history * seq_length)
    future_seq_len = seq_length - history_seq_len

    cond_dim = args.cond_dim
    agent_traj_history_encoder = SetTransformer(dim_input=2*history_seq_len, num_outputs=1, dim_output=cond_dim)

    model = Unet1D(
        dim = 64,
        cond_dim = args.cond_dim,
        cond_encoder=agent_traj_history_encoder,
        channels = channels,
        cond_drop_prob=0.05
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = future_seq_len,
        timesteps = args.timesteps,
        sampling_timesteps = args.sampling_timesteps,
        loss_type = args.loss_type,
        beta_schedule = args.beta_schedule
    )

    try:
        train(
            diffusion,
            dataset,
            output_folder,
            batch_size = 32,
            lr = 8e-5,
            num_train_steps = args.train_steps,
        )
    except Exception as e:
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
