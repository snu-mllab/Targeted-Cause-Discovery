import os
import math
import torch
import numpy as np
from collections import defaultdict

from .find_cause import Graph
from .utils import standardize_count, torch_dtype


def load_test_data(args, index=0):
    """ Load data for testing

    Args:
        index (int, optional): dataset environment index (e.g., seed). Defaults to 0.

    Returns:
        graph (torch.tensor, optional): ground-truth cause matrix (n_var x n_var).
                                        each row indicates causes of a variable.
        inputs (tuple): containing data (n_obs x n_var), intervention matrix, and regime
    """
    if args.data_file == "custom":
        ##### Make this part ########################################
        data = None  # n_obs x n_var (torch.tensor)
        intv = None  # n_obs x n_var (torch.tensor, torch.bool)
        graph = None  # n_var x n_var (np.array) causal graph (optional)
        ############################################################

        if graph is not None:
            graph = Graph(graph).causeMatrix()  # convert direct cause to cause matrix
            graph = torch.tensor(graph, dtype=torch.int8).permute(1, 0)  # each row is causes

    elif args.data_file == "human":
        data, intv = load_human_cell(args)
        graph = None
    elif "sergio" in args.data_file:
        graph, data, intv = load_simulation_data(args, args.data_file, index, args.data_level)
    else:
        graph, data, intv = load_synthetic_data(args, args.data_file, (index + 1) * 10)

    if graph is not None:
        graph.fill_diagonal_(-100)

    regime = make_regime(intv)
    inputs = (data, intv, regime)

    print(f"Data: {data.shape}, obs: {(intv.sum(1)==0).sum()}, intv_gene: {(intv.sum(0)>0).sum()}")
    return graph, inputs


def load_human_cell(args):
    path = os.path.join(args.data_path, "k562")

    # data matrix
    if "impute" in args.data_level:
        data = np.load(os.path.join(path, "data_th10_impute.npy"))
        print("Impute!", flush=True)
    else:
        data = np.load(os.path.join(path, "data_th10.npy"))

    # intervention matrix
    intv = np.load(os.path.join(path, "intv_th10.npy"))

    # normalization
    data = standardize_count(data, shift=args.shift)
    data = torch.tensor(data, dtype=torch_dtype(args.dtype))
    intv = torch.tensor(intv)

    # select genes that have interventional data
    valid_gene = intv.sum(0) > 0
    data = data[:, valid_gene]
    intv = intv[:, valid_gene]

    # balancing obervational and interventional data
    obs_idx = intv.sum(1) == 0
    data_obs, intv_obs = data[obs_idx], intv[obs_idx]
    data_intv, intv_intv = data[~obs_idx], intv[~obs_idx]

    intv_per_gene = math.ceil((~obs_idx).sum() / valid_gene.sum())
    no_intv_size = int(1000 * intv_per_gene * args.obs_ratio)

    data_obs = data_obs[:no_intv_size]
    intv_obs = intv_obs[:no_intv_size]
    data = torch.cat([data_obs, data_intv])
    intv = torch.cat([intv_obs, intv_intv])

    return data, intv


def load_synthetic_data(args, data_file, idx):
    # load causal graph and convert to cause matrix
    path_graph = os.path.join(data_file, f"DAG{idx}.npy")
    graph = np.load(path_graph)
    graph = Graph(graph).causeMatrix()  # convert direct cause to cause matrix
    graph = torch.tensor(graph, dtype=torch.int8).permute(1, 0)  # each row is causes

    # data matrix
    path_data = os.path.join(data_file, f"data_interv{idx}.npy")
    data = np.load(path_data)
    data = torch.tensor(data, dtype=torch_dtype(args.dtype))

    if args.shift:
        data = data + args.shift

    # intervention matrix
    intv = torch.load(os.path.join(data_file, f"intv{idx}.pt"))

    return graph, data, intv


def load_simulation_data(args, data_file, idx, data_level):
    # load causal graph
    path_graph = os.path.join(data_file, f"{idx}/DAG.npy")
    graph = np.load(path_graph)

    graph = Graph(graph).causeMatrix()  # convert direct cause to cause matrix
    graph = torch.tensor(graph, dtype=torch.int8).permute(1, 0)  # each row is causes

    # data matrix
    path_data_intv = os.path.join(data_file, f"{idx}/{data_level}_intv.npy")
    data_intv = np.load(path_data_intv)

    assert args.obs_ratio > 0
    no_intv_size = int(10000 * args.obs_ratio)
    no_intv_size = int(no_intv_size * (len(data_intv) // graph.shape[-1] / 10))

    path_data = os.path.join(data_file, f"{idx}/{data_level}.npy")
    data_obs = np.load(path_data)[:no_intv_size]

    data = np.concatenate([data_intv, data_obs], axis=0)

    # normalization
    data = standardize_count(
        data,
        log=args.log,
        shift=args.shift,
    )
    data = torch.tensor(data, dtype=torch_dtype(args.dtype))

    assert torch.isfinite(data).all(), print(data_file)

    # intervention matrix
    intv = np.load(os.path.join(data_file, f"{idx}/intv.npy"))
    intv_obs = np.zeros(data_obs.shape, dtype=intv.dtype)

    intv = np.concatenate([intv, intv_obs], axis=0)
    intv = torch.tensor(intv)

    return graph, data, intv


def make_regime(intv):
    """ create a mapping for interventions
    """
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
