import argparse
import logging
from pathlib import Path
from datetime import datetime
from diff_traj.dataset.dataset import StateDataset
from diff_traj.baselines.fcnet import FCNet, Trainer
from diff_traj.cfg import cfg
from diff_traj.utils.logs import setup_logging
from diff_traj.utils.io import write_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-l', '--layers', nargs="+", type=int, help="Enter layer sizes like -s 1 2 3")
    parser.add_argument('-n', '--train_steps', type=int, default=100000)
    parser.add_argument('-d', '--dataset_folder', default='./data/subset/')
    parser.add_argument('-o', '--output_folder', default='./results/demo/')
    parser.add_argument('-ll', '--log_level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok = True)
    now = datetime.now().strftime("%b-%d-%H-%M-%S")
    setup_logging(args.log_level, True, output_folder/f"train-fc-{now}.log")

    dataset = StateDataset(cfg, args.dataset_folder)
    logging.info('loaded dataset')

    if args.layers is not None:
        layers = args.layers
    else:
        layers = [cfg.params_length, 64, 64*2, 64*4, cfg.traj_length]

    model = FCNet(layers)

    logging.info('built fcnet')

    trainer = Trainer(
        model,
        dataset,
        cfg,
        output_folder,
        train_batch_size = 32,
        train_lr = 3e-4,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,              # gradient accumulation steps
        ema_decay = 0.995,                          # exponential moving average decay
    )

    logging.info('built trainer')
    for checkpoint in [1, 10, 20, 30, 40, 50, 60,70,80,90,99,'finished']:
        trainer.load(f'{args.model_path}/model-{checkpoint}.pt')
        for i, obj in enumerate(trainer.sample(dataset), 1):
            write_obj(obj, output_folder/f'{checkpoint}-sampled-{i}.pkl')
        logging.info('finished logging checkpoint %s', checkpoint)

    logging.info("finished training")

if __name__ == '__main__':
    main()
