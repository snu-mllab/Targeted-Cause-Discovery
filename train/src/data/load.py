"""
DataModule is entrypoint into all data-related objects.

We create different datasets via diamond inheritence
which is arguably horrible and clever hehe.

"""

import os
from functools import partial
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset import TrainDataset, TestDataset
from .dataset_ensemble import TestDatasetEnsemble
from .find_cause import Graph
from .utils import collate, Logger, standardize_count

import time

logger = Logger(__name__)


def torch_dtype(dtype):
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16


def load_synthetic_data(args, data_file, idx, save=False):
    # Graph
    path_graph = os.path.join(data_file, f"DAG{idx}.npy")
    graph = np.load(path_graph)
    if save:
        np.save(path_graph, graph.astype(bool))
    if args.cause:
        graph = Graph(graph).causeMatrix()
    graph = torch.tensor(graph, dtype=torch.int8).permute(1, 0)  # each row is causes

    # Observation
    path_data = os.path.join(data_file, f"data_interv{idx}.npy")
    data = np.load(path_data)
    if save:
        np.save(path_data, data.astype(np.float32))
    data = torch.tensor(data, dtype=torch_dtype(args.dtype))

    if args.shift:
        data = data + args.shift

    # Intervention
    file = os.path.join(data_file, f"regime{idx}.pt")
    if save or not os.path.exists(file):
        intv, regime = make_regime_from_csv(os.path.join(data_file, f"intervention{idx}.csv"), data)

        torch.save(intv, os.path.join(data_file, f"intv{idx}.pt"))
        torch.save(regime, file)
    else:
        intv = torch.load(os.path.join(data_file, f"intv{idx}.pt"))
        regime = torch.load(file)

    return graph, data, intv, regime


def make_regime_from_csv(csv_path, data):
    regime_to_intv = {}
    regime_obs = defaultdict(list)  # regime -> obs index list
    len_to_regime = defaultdict(list)
    intv_to_regime = {}
    intv = torch.zeros(data.shape, dtype=bool)

    with open(csv_path) as f:
        for k, line in enumerate(f.readlines()):
            line = line.strip()

            if len(line) > 0:
                # insert to intv matrix
                line_idx = [int(l) for l in line.split(',')]
                intv[k, line_idx] = True
            else:
                line = "obs"
                line_idx = []

            if line not in intv_to_regime:
                regime = len(intv_to_regime)

                intv_to_regime[line] = regime
                len_to_regime[len(line_idx)].append(regime)
                regime_to_intv[regime] = set(line_idx)

            regime_obs[intv_to_regime[line]].append(k)

    regime = {
        "regime_to_intv": regime_to_intv,
        "regime_obs": regime_obs,
        "len_to_regime": len_to_regime
    }
    return intv, regime


def make_regime(intv):
    regime_to_intv = {}
    regime_obs = defaultdict(list)  # regime -> obs index list
    len_to_regime = defaultdict(list)
    intv_to_regime = {}

    for k, idx_ in enumerate(intv):
        line_idx = idx_.nonzero(as_tuple=True)[0].tolist()

        if len(line_idx) == 0:
            line = "obs"
        else:
            line = tuple(line_idx)

        if line not in intv_to_regime:
            regime = len(intv_to_regime)

            intv_to_regime[line] = regime
            len_to_regime[len(line_idx)].append(regime)
            regime_to_intv[regime] = set(line_idx)

        regime_obs[intv_to_regime[line]].append(k)

    regime = {
        "regime_to_intv": regime_to_intv,
        "regime_obs": regime_obs,
        "len_to_regime": len_to_regime
    }
    return regime


def load_simulation_data(args, data_file, idx, data_level, save=False):
    # Graph
    path_graph = os.path.join(data_file, f"{idx}/DAG.npy")
    graph = np.load(path_graph)
    if args.cause:
        graph = Graph(graph).causeMatrix()
    graph = torch.tensor(graph, dtype=torch.int8).permute(1, 0)  # each row is causes

    # Observation
    path_data_intv = os.path.join(data_file, f"{idx}/{data_level}_intv.npy")
    data_intv = np.load(path_data_intv)

    assert args.obs_ratio > 0
    no_intv_size = int(10000 * args.obs_ratio)
    no_intv_size = int(no_intv_size * (len(data_intv) // graph.shape[-1] / 10))

    path_data = os.path.join(data_file, f"{idx}/{data_level}.npy")
    data_obs = np.load(path_data)[:no_intv_size]

    data = np.concatenate([data_intv, data_obs], axis=0)
    data = standardize_count(
        data,
        log=args.log,
        shift=args.shift,
    )
    data = torch.tensor(data, dtype=torch_dtype(args.dtype))

    assert torch.isfinite(data).all(), print(data_file)

    # Intervention
    intv = np.load(os.path.join(data_file, f"{idx}/intv.npy"))
    intv_obs = np.zeros(data_obs.shape, dtype=intv.dtype)

    intv = np.concatenate([intv, intv_obs], axis=0)
    intv = torch.tensor(intv)

    # Regime
    file = os.path.join(data_file, f"{idx}/regime_{no_intv_size}.pt")
    if save or not os.path.exists(file):
        regime = make_regime(intv)
        torch.save(regime, file)
    else:
        # You should process priorly
        regime = torch.load(file)

    return graph, data, intv, regime


def load_simulation_data_obs(args, data_file, idx, data_level, save=None):
    # Graph
    path_graph = os.path.join(data_file, f"{idx}/DAG.npy")
    graph = np.load(path_graph)
    if args.cause:
        graph = Graph(graph).causeMatrix()
    graph = torch.tensor(graph, dtype=torch.int8).permute(1, 0)  # each row is causes

    # Observation
    no_intv_size = int(10000 * args.obs_ratio)
    path_data = os.path.join(data_file, f"{idx}/{data_level}.npy")
    data = np.load(path_data)[:no_intv_size]

    data = standardize_count(
        data,
        log=args.log,
        libnorm=args.libnorm,
        atol=args.atol,
        n_quant=args.n_quant,
        shift=args.shift,
    )
    data = torch.tensor(data, dtype=torch_dtype(args.dtype))
    assert torch.isfinite(data).all(), print(data_file)

    # Intervention
    intv = torch.zeros(data.shape, dtype=torch.bool)

    # Regime
    regime = None

    return graph, data, intv, regime


class DataModule(pl.LightningDataModule):

    def __init__(self, args, verbose=True, save=False, only_pred=False, ensemble=False):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        graph_list = []
        data_list = []
        intv_list = []
        regime_list = []

        # Load data
        self.data_file = args.data_file
        for data_file in self.data_file:
            if "synthetic" in data_file:
                logger.printt(f"load data from {data_file}")
                for idx in range(args.n_env):
                    if only_pred:
                        if idx % 10 != 9:
                            graph_list.append(None)
                            data_list.append(None)
                            intv_list.append(None)
                            regime_list.append(None)
                            continue

                    graph, data, intv, regime = load_synthetic_data(args,
                                                                    data_file,
                                                                    idx + 1,
                                                                    save=save)

                    graph_list.append(graph)
                    data_list.append(data)
                    intv_list.append(intv)
                    regime_list.append(regime)

            elif "sergio" in data_file:
                logger.printt(f"load data from {data_file} {args.data_level}")
                for data_level in args.data_level:
                    for idx in range(args.n_env):
                        if only_pred:
                            if idx % 10 != 9:
                                graph_list.append(None)
                                data_list.append(None)
                                intv_list.append(None)
                                regime_list.append(None)
                                continue

                        if ("ecoli" in data_file) or ("yeast" in data_file):
                            idx = idx // 10

                        if args.only_obs:
                            load_fn = load_simulation_data_obs
                        else:
                            load_fn = load_simulation_data

                        graph, data, intv, regime = load_fn(args,
                                                            data_file,
                                                            idx,
                                                            data_level,
                                                            save=save)

                        graph_list.append(graph)
                        data_list.append(data)
                        intv_list.append(intv)
                        regime_list.append(regime)

        # Split data
        n = len(data_list)
        split_train = [i for i in range(n) if i % 10 <= 7]
        split_val = [i for i in range(n) if i % 10 == 8]
        split_test = [i for i in range(n) if i % 10 == 9]

        if not only_pred:
            self.train = TrainDataset(graph_list,
                                      data_list,
                                      intv_list,
                                      regime_list,
                                      split=split_train,
                                      obs_size=args.obs_size,
                                      node_size=args.num_vars,
                                      only_obs=args.only_obs)
            self.val = TestDataset(graph_list,
                                   data_list,
                                   intv_list,
                                   regime_list,
                                   split=split_val,
                                   obs_size=args.obs_size,
                                   node_size=args.num_vars,
                                   only_obs=args.only_obs)

        dataset_fn = TestDatasetEnsemble if ensemble else TestDataset
        self.test = dataset_fn(graph_list,
                               data_list,
                               intv_list,
                               regime_list,
                               split=split_test,
                               obs_size=args.obs_size,
                               node_size=args.num_vars,
                               only_obs=args.only_obs)

        if verbose and not only_pred:
            logger.print(
                f"- {'Env':5s}: train {len(self.train.data)}, val {len(self.val.data)}, test {len(self.test.data)}"
            )
            logger.print(
                f"- {'Data':5s}: train {len(self.train)}, val {len(self.val)}, test {len(self.test)}"
            )
            logger.print(
                f"- {'Obs':5s}: train {self.train.obs_size}, val {self.val.obs_size}, test {self.test.obs_size}"
            )

    def train_dataloader(self):
        train_loader = DataLoader(self.train,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=(not self.args.debug),
                                  collate_fn=partial(collate, self.args))
        return train_loader

    def val_dataloader(self):
        # batch_size smaller since we sample more batches on average
        val_loader = DataLoader(self.val,
                                batch_size=max(self.batch_size // 4, 1),
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=(not self.args.debug),
                                collate_fn=partial(collate, self.args))
        return val_loader

    def predict_dataloader(self):
        test_loader = DataLoader(self.test,
                                 batch_size=max(self.batch_size // 4, 1),
                                 num_workers=self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return test_loader

    def test_dataloader(self):
        return self.predict_dataloader()
