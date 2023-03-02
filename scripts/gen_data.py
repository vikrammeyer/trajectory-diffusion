import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
import logging
import pathlib
from datetime import datetime
from trajdiff.utils import setup_logging, set_seed, write_metadata, write_obj
from trajdiff.dataset.generator import gen_samples
from trajdiff import cfg

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
        default=42,
        type=int
    )
    parser.add_argument(
        "-c",
        "--constrain_obsts",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    now = datetime.now().strftime("%b-%d-%H-%M-%S")
    output_folder = pathlib.Path(args.output_folder)

    if not output_folder.is_dir():
        os.mkdir(output_folder)

    write_metadata(cfg, output_folder)

    setup_logging(args.log_level, True, output_folder/f"data-gen-{now}.log")

    # n_per_process = args.n_samples
    timeout = 25 * args.n_per_file
    n_jobs = int(args.n_samples / args.n_per_file)
    # starter seed to create seeds for each function

    set_seed(args.seed)
    seeds = [random.random() * (i + 1) for i in range(n_jobs)]

    with ProcessPoolExecutor(max_workers=args.processes) as executor:

        futures = [executor.submit(gen_samples, cfg, args.n_per_file, seed, args.constrain_obsts) for seed in seeds]

        for i, future in enumerate(futures, 1):
            try:
                results = future.result(timeout=timeout)

                write_obj(results, pathlib.Path(f"{args.output_folder}/chunk{i}.pkl"))

            except Exception as e:
                logging.error("Error with generating a chunk of problems: ", e)

if __name__ == '__main__':
    main()
