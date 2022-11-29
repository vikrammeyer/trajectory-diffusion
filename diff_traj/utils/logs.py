import sys
import logging
import pathlib
from typing import Optional

def setup_logging(level: str, print_stdout: bool, filename: Optional[pathlib.Path] = None):
    fmt = '%(asctime)s | %(levelname)8s | %(message)s'

    handlers = []

    if print_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))

    levels = {"DEBUG": logging.DEBUG,
              "INFO": logging.INFO,
              "WARNING": logging.WARNING,
              "ERROR": logging.ERROR,
              "CRITICAL": logging.CRITICAL}

    logging.basicConfig(format=fmt, level=levels[level.upper()], handlers=handlers)