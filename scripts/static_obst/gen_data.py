import argparse
import random
from concurrent.futures import ProcessPoolExecutor
import logging
from trajdiff.utils import setup, set_seed, write_metadata, write_obj
from trajdiff.static_obst.generator import gen_samples
from trajdiff.static_obst import cfg


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
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-c", "--constrain_obsts", default=False, action="store_true")
    args = parser.parse_args()

    output_folder = setup(args)

    write_metadata(cfg, output_folder)

    timeout = 25 * args.n_per_file
    n_jobs = int(args.n_samples / args.n_per_file)

    # starter seed to create seeds for each function
    set_seed(args.seed)
    seeds = [random.random() * (i + 1) for i in range(n_jobs)]

    with ProcessPoolExecutor(max_workers=args.processes) as executor:

        futures = [
            executor.submit(
                gen_samples, cfg, args.n_per_file, seed, args.constrain_obsts
            )
            for seed in seeds
        ]

        for i, future in enumerate(futures, 1):
            try:
                results = future.result(timeout=timeout)

                write_obj(results, args.output_folder / f"chunk{i}.pkl")
                logging.info("Generated chunk %d", i)

            except Exception as e:
                logging.error("Error with generating a chunk of problems: ", e)


if __name__ == "__main__":
    main()
