import numpy as np
import torch
from math import ceil
from data import collate
from datetime import datetime


def sample_node(n_node, target_node, chunk_size, idx, en_idx):
    """ Subsample variables for ensembled inference
    """
    node_left = np.ones(n_node, dtype=bool)
    node_left[target_node] = False
    node_left = np.arange(n_node)[node_left].tolist()

    cand = []
    for i in range(en_idx):
        cand.extend(node_left[i::en_idx])

    offset = max(0, chunk_size * (idx + 1) - len(cand))
    other_node = cand[chunk_size * idx - offset:chunk_size * (idx + 1) - offset]

    if type(target_node) == int:
        target_node = [target_node]
    indices = target_node + other_node
    return indices


def sample_data(args, inputs, offset_obs, node_indices):
    """ Subsample observations for ensembled inference
    """
    full_data, full_intv, regime = inputs

    regime_slct = [regime["len_to_regime"][0][0]] + node_indices

    idx_obs_pool = []
    for r in regime_slct:
        idx_obs_pool.extend(regime["regime_obs"][r])

    # sample observation
    hop = ceil(len(idx_obs_pool) / args.obs_size)
    idx_obs = np.arange(args.obs_size) * hop + offset_obs
    idx_obs = idx_obs % len(idx_obs_pool)
    idx_obs = np.array(idx_obs_pool)[idx_obs]

    data, intv = full_data[idx_obs], full_intv[idx_obs]

    # sample node
    idx_node = torch.tensor(node_indices)
    data = data[:, idx_node]
    intv = intv[:, idx_node]

    return {"data": data, "intv": intv, "idx_node": idx_node}


def set_anchor(target, logit, count, args):
    # We empirically observed that using the fixed set of variables as anchors
    # during ensembled inference enhances the performance (1~5%).
    # We select the top variables that have the highest cause score values up to the current iteration.
    score = logit / (count.float() + 1e-8)
    score[target] = -100
    indices = score.topk(args.anchor_size)[1]
    indices = indices.tolist()

    indices = [target] + indices
    return indices


def infer(args, model, inputs, targets=None):
    """ calculate cause scores for targets

    Args:
        inputs (tuple): tuple containing data (n_obs x n_var), intervention, and regime
        targets (list, optional): Target indices. Defaults to None.

    Returns:
        pred_all (torch.tensor, n_target x n_var): Cause scores for targets
    """

    pred_all = []
    n_node = inputs[0].shape[-1]
    if targets is None:
        targets = np.arange(n_node)

    for i, target_node in enumerate(targets):
        if i % 100 == 0:
            print(f"{target_node} {datetime.now().strftime('%H:%M')}", flush=True)

        if target_node >= n_node:
            continue
        full_logit = torch.zeros(n_node, dtype=torch.float32, device="cuda")
        count = torch.zeros(n_node, dtype=torch.int, device="cuda")

        for en_idx in range(1, args.en_vars + 1):
            anchors = set_anchor(target_node, full_logit, count, args)

            chunk_size = args.num_vars - len(anchors)
            n_chunk = ceil((n_node - len(anchors)) / chunk_size)
            for offset_node in range(n_chunk):
                # Sample node
                batch = []
                node_indices = sample_node(n_node, anchors, chunk_size, offset_node, en_idx)

                # Sample batch
                for offset_obs in range(args.en_obs):
                    example = sample_data(args, inputs, offset_obs, node_indices)
                    batch.append(example)

                idx_node = example["idx_node"]
                batch = collate(args, batch, keys=["data", "intv"])
                for k, v in batch.items():
                    batch[k] = v.cuda()

                # Forward
                with torch.no_grad():
                    x = model.encoder(batch)
                    logit = model.top_layer(x)
                    logit = logit.mean(0)

                # Aggregate prediction
                assert idx_node[0] == target_node
                full_logit[idx_node] += logit[0]
                count[idx_node] += 1

        full_logit = full_logit / count.float()
        pred_all.append(full_logit.cpu())

    pred_all = torch.stack(pred_all, dim=0)
    return pred_all
