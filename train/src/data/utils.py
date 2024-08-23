import os
import warnings
import numpy as np
import torch
import logging

from torch.utils.data import default_collate

from datetime import datetime


def collate(args, batch, keys=["data", "intv", "label"]):
    """
        Overwrite default_collate for jagged tensors
    """
    # initialize new batch
    # and skip invalid items haha
    batch = {key: [item[key] for item in batch if key in item] for key in keys}
    new_batch = {}
    for key, val in batch.items():
        if not torch.is_tensor(val[0]) or val[0].ndim == 0:
            new_batch[key] = default_collate(val)
        else:
            new_batch[key] = torch.stack(val, dim=0)

    return new_batch


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


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


def _np_cpm_standardizer(x, libnorm=True, atol=1e-8):
    # compute library sizes (sum of row)
    x_libsize = x.sum(-1, keepdims=True)
    if not libnorm:
        x_libsize = x_libsize.mean()

    # divide each cell by library size and multiply by 10^6
    # will yield nan for rows with zero expression and for zero expression entries
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log2cpm = np.where(np.isclose(x, 0.0, atol=atol), np.nan, np.log2(x / (x_libsize * 1e-6)))
    return log2cpm


def _np_cpm_shift_scale(x, shift, scale):
    # shift and scale
    x = (x - np.where(np.isnan(shift), 0.0, shift)) / np.where(np.isnan(scale), 1.0, scale)

    # set nans (set for all-zero rows, i.e. zero libsize) to minimum (i.e. to zero)
    # catch empty arrays (when x_int is empty and has axis n=0)
    if not x.size == 0:
        x = np.where(np.isnan(x), 0.0, x)

    return x


def standardize_count(x, log=True, libnorm=True, atol=1e-8, shift=None):
    """
    log2 CPM normalization for gene expression count data
    https://bioconductor.org/packages/release/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf
    https://rdrr.io/bioc/edgeR/src/R/cpm.R
    http://luisvalesilva.com/datasimple/rna-seq_units.html

    log2 CPM(mat) = log2( 10^6 * mat / libsize ) where libsize = mat.sum(1) (i.e. sum over genes)

    Why log2(1+cpm) is a bad idea https://support.bioconductor.org/p/107719/
    Specific scaling https://support.bioconductor.org/p/59846/#59917

    """
    if log:
        # cpm normalization
        x = _np_cpm_standardizer(x, libnorm=libnorm, atol=atol)

    # subtract min (~robust global median) and divide by global std dev
    if shift is not None:
        global_mean = np.nanmean(x, axis=(-1, -2), keepdims=True)
        global_std = np.nanstd(x, axis=(-1, -2), keepdims=True)

        x = (x - global_mean) / global_std
        x = x + shift

        x = np.where(np.isnan(x), 0.0, x)
        x = np.maximum(x, 0.0)

    else:
        global_min = np.nanmin(x, axis=(-1, -2), keepdims=True)
        global_std = np.nanstd(x, axis=(-1, -2), keepdims=True)
        if not libnorm:
            global_min -= 1

        x = _np_cpm_shift_scale(x, global_min, global_std)

    return x
