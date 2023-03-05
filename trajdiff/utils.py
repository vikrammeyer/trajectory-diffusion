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
    fmt = "%(asctime)s | %(levelname)8s | %(message)s"

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

    logging.basicConfig(format=fmt, level=levels[level.upper()], handlers=handlers)

    hash = git_hash()
    logging.info("git hash: %s", hash)


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


def write_metadata(cfg, dataset_folder: pathlib.Path):
    """
    Writes the configuration dictionary to a metadata file in JSON format.

    Args:
        cfg (dict): The configuration dictionary to be written.
        dataset_folder (pathlib.Path): The folder where the metadata file should be written.
    """
    meta_file = dataset_folder / "metadata.json"
    metadata = json.dumps(cfg.__dict__)
    meta_file.write_text(metadata)


def git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
