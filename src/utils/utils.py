import os
import enum
import logging
import numpy as np

# Global logger
logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ActionMode(enum.Enum):
    STUDENT = 1
    TEACHER = 2


class PlotMode(enum.Enum):
    POSITION = 1
    VELOCITY = 2


class TruncatePathFormatter(logging.Formatter):
    def format(self, record):
        cwd = os.getcwd()
        # print(f"pathname: {record.pathname}")
        if cwd not in record.pathname:
            return super().format(record)
        else:
            pathname = record.pathname.split(f'{cwd}/')[1]
            record.pathname = pathname
            return super().format(record)


def energy_value(state: np.ndarray, p_mat: np.ndarray) -> int:
    """
    Get system energy value represented by s^T @ P @ s
    """
    return state.transpose() @ p_mat @ state


def get_discrete_Ad_Bd(Ac: np.ndarray, Bc: np.ndarray, T: int):
    """
    Get the discrete form of matrices Ac and Bc given the sample period T
    """
    Ad = Ac * T + np.eye(4)
    Bd = Bc * T
    return Ad, Bd


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"{dir_path} does not exist, creating...")


def is_dir_empty(path):
    """
    Check a directory is empty or not
    """
    # List the contents of the directory
    contents = os.listdir(path)
    return len(contents) == 0


def logging_mode(mode: str):
    if mode == 'DEBUG':
        return logging.DEBUG
    elif mode == 'INFO':
        return logging.INFO
    elif mode == 'WARNING':
        return logging.WARNING
    elif mode == 'ERROR':
        return logging.ERROR
    elif mode == 'CRITICAL':
        return logging.CRITICAL
    elif mode is None:
        return logging.CRITICAL + 1
    else:
        raise RuntimeError(f"Unrecognized logging mode: {mode}")
