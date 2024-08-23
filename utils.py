import random
import numpy as np
import torch

from datetime import datetime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def printt(*args, **kwargs):
    print(f"[{get_timestamp()}]", *args, **kwargs)
