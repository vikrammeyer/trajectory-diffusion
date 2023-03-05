import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
import logging
import pathlib
from trajdiff.utils import setup_logging, set_seed, write_obj
from trajdiff.multiagent import generate_multiple_trajs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        required=True,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "-a",
        "--n_agents",
        type=int,
        default=15,
    )
    parser.add_argument(
        "-t",
        "--traj_len",
        type=int,
        default=1000
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        help="Number of processes to generate data in parallel (defaults to all cores)",
        default=None,
    )
    parser.add_argument("-f", "--n_per_file", type=int, default=1000)
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="Output folder for files from each process",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int
    )
    args = parser.parse_args()

    output_folder = pathlib.Path(f'results/{args.output_folder}')

    if not output_folder.is_dir():
        os.mkdir(output_folder)

    setup_logging(args.log_level, True, output_folder/f"multiagent-data-gen.log")

    timeout = 25 * args.n_per_file
    n_jobs = int(args.n_samples / args.n_per_file)

    # starter seed to create seeds for each function
    set_seed(args.seed)
    seeds = [random.random() * (i + 1) for i in range(n_jobs)]

    with ProcessPoolExecutor(max_workers=args.processes) as executor:

        futures = [executor.submit(generate_multiple_trajs, seed, args.n_per_file, args.n_agents, args.traj_len) for seed in seeds]

        for i, future in enumerate(futures, 1):
            try:
                data = future.result(timeout=timeout)
                write_obj(data, output_folder/f"chunk{i}.pkl")
                logging.info('Generated chunk %d', i)
            except Exception as e:
                logging.error("Error with generating a chunk of problems: ", e)

if __name__ == '__main__':
    main()
