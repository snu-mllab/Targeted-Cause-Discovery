import os
import csv
import pickle
import logging
import random
import numpy as np
import torch

from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_pickle(fp, data):
    with open(fp, "wb+") as f:
        pickle.dump(data, f)


def read_pickle(fp):
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data


def read_csv(fp, fieldnames=None, delimiter=',', str_keys=[]):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        # iterate and append
        for item in reader:
            data.append(item)
    return data


# -------- general


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def printt(*args, **kwargs):
    print(f"[{get_timestamp()}]", *args, **kwargs)


def get_suffix(metric):
    suffix = "model_best_"
    suffix += "step{step}_epoch{epoch}_{"
    suffix += metric + ":.3f}"
    return suffix


class Logger():

    def __init__(self, name="", file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if file is not None:
            handler = logging.FileHandler(file, mode="w")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def print(self, msg=""):
        self.logger.info(msg)

    def printt(self, msg=""):
        if len(msg) > 0:
            if msg[0] == "\n":
                self.logger.info(f"\n[{get_timestamp()}] {msg[1:]}")
            else:
                self.logger.info(f"[{get_timestamp()}] {msg}")
        else:
            self.logger.info(f"[{get_timestamp()}]")
