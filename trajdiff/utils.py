import json
import logging
import os
import pathlib
import pickle
import random
import subprocess
import sys
from typing import Optional

import numpy as np
import torch


def setup(args, log_file, print_stdout=True) -> pathlib.Path:
    """
    Create output folder, setup logging, log the git hash, log the script args

    Args:
        args (Namespace): An object containing the command-line arguments passed to the script.
                                Must contain `output_folder` and `log_level`.

    Returns:
        A `Path` object representing the output folder.
    """
    output_folder = pathlib.Path(args.output_folder)
    output_folder_exists = output_folder.is_dir()

    if not output_folder_exists:
       output_folder.mkdir()

    setup_logging(args.log_level, print_stdout=print_stdout, filename=output_folder / log_file)

    # Log the git commit for reproducability
    hash = git_hash()
    logging.info("git hash: %s", hash)

    # Log the arguments used to run the script for reproducability
    logging.info(args)

    # To help determine when output files get mixed together from different runs
    if output_folder_exists:
        logging.warning(
            "output folder %s already exists...saving new outputs to it", output_folder
        )

    return output_folder


def set_seed(seed: int):
    """
    Sets the random seed for numpy, torch, and CUDA (if available) to ensure reproducibility.

    Args:
        seed (int): The random seed to be used. (np requires integer seed)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(
    level: str, print_stdout: bool, filename: Optional[pathlib.Path] = None
):
    """
    Sets up logging to write log messages to stdout and/or a file.

    Args:
        level (str): The logging level (one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        print_stdout (bool): Whether to print log messages to stdout.
        filename (Optional[pathlib.Path]): The file to write log messages to (if specified).
    """
    handlers = []

    if print_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))

    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    fmt = "%(asctime)s|%(levelname)s| %(message)s"
    logging.basicConfig(
        format=fmt,
        datefmt="%m-%d %H:%M:%S",
        level=levels[level.upper()],
        handlers=handlers,
    )


def write_obj(obj, filename):
    """
    Writes a Python object to a file using pickle.

    Args:
        obj: The object to be written.
        filename (str): The path to the file to write to.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def read_file(filename):
    """
    Reads a Python object from a file using pickle.

    Args:
        filename (str): The path to the file to read from.

    Returns:
        The Python object read from the file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_dict(dct, filename: pathlib.Path):
    """
    Write a dictionary to json file pretty printed.

    Args:
        dct (dict): A dictionary to be written to the file.
        filename (pathlib.Path): A path representing the file to write to.
    """
    filename.write_text(json.dumps(dct, indent=4))


def write_metadata(cfg, dataset_folder: pathlib.Path):
    """
    Writes the configuration dictionary to a metadata file in JSON format.

    Args:
        cfg (dict): The configuration dictionary to be written.
        dataset_folder (pathlib.Path): The folder where the metadata file should be written.
    """
    write_dict(cfg.__dict__, dataset_folder / "metadata.json")


def git_hash() -> str:
    """
    Get the git hash. Assumes being run in a git repo (otherwise error).

    Returns:
        A str representing current git hash.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
