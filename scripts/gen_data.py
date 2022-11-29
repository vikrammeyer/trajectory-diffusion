import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
import logging
import pathlib
from datetime import datetime

from diff_traj.utils.logs import setup_logging
from diff_traj.dataset.generator import gen_samples
from diff_traj.cfg import cfg
from diff_traj.utils.repro import set_seed
import diff_traj.utils.io as io

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        required=True,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        help="Number of processes to generate data in parallel (defaults to all cores)",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="Output folder for .npy files from each process",
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
        default=42
    )
    args = parser.parse_args()

    now = datetime.now().strftime("%b-%d-%H-%M-%S")
    output_folder = pathlib.Path(args.output_folder)

    if not output_folder.is_dir():
        os.mkdir(output_folder)

    io.write_metadata(cfg, output_folder)

    setup_logging(args.log_level, True, output_folder/f"data-gen-{now}.log")

    timeout = 25 * args.n_per_file

    # starter seed to create seeds for each function

    set_seed(args.seed)
    seeds = [random.random() * (i + 1) for i in range(args.processes)]

    with ProcessPoolExecutor(max_workers=args.processes) as executor:

        futures = [executor.submit(gen_samples, cfg, args.n_per_file, seed) for seed in seeds]

        for i, future in enumerate(futures, 1):
            try:
                results = future.result(timeout=timeout)

                io.write_obj(results, pathlib.Path(f"{args.output_folder}/chunk{i}.pkl"))

            except Exception as e:
                logging.error("Error with generating a chunk of problems: ", e)
