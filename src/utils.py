import numpy as np
import logging
import os


def map_labels(label_arr):
    """Map raw BraTS labels (0-3) to binary ET/TC/WT masks.

    ET  = label 3
    TC  = label 2 | 3
    WT  = label 1 | 2 | 3
    """
    et = (label_arr == 3).astype(np.uint8)
    tc = ((label_arr == 2) | (label_arr == 3)).astype(np.uint8)
    wt = ((label_arr >= 1) & (label_arr <= 3)).astype(np.uint8)
    return {'ET': et, 'TC': tc, 'WT': wt}


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
