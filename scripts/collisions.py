import argparse
import logging
from trajdiff.utils import read_file, write_dict, setup
from trajdiff.multiagent import calc_collisions

import glob
import statistics
from pathlib import Path

def calc_metrics(collisions):
    return {
    'mean': statistics.mean(collisions),
    'stdev': statistics.stdev(collisions),
    'mode': statistics.mode(collisions),
    'median': statistics.median(collisions)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
    )
    args = parser.parse_args()

    output_folder = setup(args, 'calc_collisions.log')

    collisions = []
    files = glob.glob(f'data/{args.input_folder}/*.pkl')
    logging.debug(files)

    for file in files:
        chunk = []
        data = read_file(file)
        logging.debug(len(data))
        for sample in data:
            traj = sample['trajectories']
            radii = sample['radii']

            n = calc_collisions(traj, radii)
            chunk.append(n)
        logging.debug('')
        metrics = calc_metrics(chunk)
        collisions.extend(chunk)

        logging.info('finished checking file %s', file)
        file = Path(file)
        write_dict(metrics, output_folder/f'metrics-{file.stem}.json')

    metrics = calc_metrics(collisions)
    write_dict(metrics, output_folder/'metrics-all-trajs.json')
    logging.info('finished calculating collisions.')

if __name__ == '__main__':
    main()