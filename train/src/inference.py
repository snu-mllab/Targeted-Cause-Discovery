"""
Main inference file. Call via inference.sh
"""
import os
from collections import defaultdict
import numpy as np
import torch

from args import parse_args, DATAPATH
from data import DataModule
from data.utils import collate
from model import load_model
from utils import Logger, set_seed
from math import ceil

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

auroc_fn = BinaryAUROC()
ap_fn = BinaryAveragePrecision()

logger = Logger(__name__)


def sample_data(args, full_data, full_intv, full_graph, regime, offset_obs, node_indices):
    regime_slct = [regime["len_to_regime"][0][0]] + node_indices
    idx_obs_pool = []
    for r in regime_slct:
        idx_obs_pool.extend(regime["regime_obs"][r])

    # sample observation
    hop = len(idx_obs_pool) // args.obs_size
    idx_obs = np.arange(args.obs_size) * hop + offset_obs
    idx_obs = idx_obs % len(idx_obs_pool)
    idx_obs = np.array(idx_obs_pool)[idx_obs]

    data, intv = full_data[idx_obs], full_intv[idx_obs]

    # sample node
    idx_node = torch.tensor(node_indices)
    data = data[:, idx_node]
    intv = intv[:, idx_node]
    label = full_graph[idx_node][:, idx_node]

    return {"data": data, "intv": intv, "label": label, "idx_node": idx_node}


def set_target(g):
    indices = torch.arange(len(g))
    n_cause = g.sum(1).cpu() + 100
    cond = n_cause >= 1
    indices = indices[cond]

    print(f"n_target: {len(indices)}")
    return indices.tolist()


def set_anchor(target, g, logit, count, args):
    score = None
    score = logit / (count.float() + 1e-8)

    if score is not None:
        score[target] = -100
        indices = score.topk(args.anchor_size)[1]
        indices = indices.tolist()

    indices = [target] + indices
    return indices


def sample_node(g, target_node, chunk_size, idx, en_idx):
    n_node = len(g)
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


def eval_model(args, model, data):
    results_all = defaultdict(list)
    pred_all, true_all = [], []

    for idx_graph in range(data.test.n_graph):
        full_graph = data.test.label[idx_graph].cuda()
        full_data = data.test.data[idx_graph]
        full_intv = data.test.intv[idx_graph]
        regime = data.test.regime[idx_graph]

        _, total_node = data.test.data[idx_graph].shape
        pred, true = [], []

        target_node_list = set_target(full_graph)
        for _, target_node in enumerate(target_node_list):
            full_logit = torch.zeros(total_node, dtype=torch.float32, device="cuda")
            count = torch.zeros(total_node, dtype=torch.int, device="cuda")

            for en_idx in range(1, args.en_vars + 1):
                anchors = set_anchor(target_node, full_graph, full_logit, count, args)

                chunk_size = args.num_vars - len(anchors)
                n_chunk = ceil((total_node - len(anchors)) / chunk_size)

                for offset_node in range(n_chunk):
                    # Sample node
                    batch = []
                    node_indices = sample_node(full_graph, anchors, chunk_size, offset_node, en_idx)

                    # Sample batch
                    for offset_obs in range(args.en_obs):
                        example = sample_data(args, full_data, full_intv, full_graph, regime,
                                              offset_obs, node_indices)
                        batch.append(example)

                    idx_node = example["idx_node"]
                    batch = collate(args, batch)
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
            gt = full_graph[target_node]

            if args.save_pred:
                pred.append(full_logit.cpu())
                true.append(gt.cpu())

            # Evaluate
            p = full_logit[gt != -100]
            t = gt[gt != -100]
            auroc = auroc_fn(p, t) * 100
            ap = ap_fn(p, t) * 100

            results_all['idx_graph'].append(idx_graph)
            results_all['target'].append(target_node)
            results_all['count'].append(t.sum().item())
            results_all['auroc'].append(auroc.item())
            results_all['ap'].append(ap.item())

            print(f"[{target_node:3d}] {auroc:5.2f}  {ap:5.2f}  {t.sum():2d}", flush=True)

        logger.print(f"{np.mean(results_all['auroc']):.2f} {np.mean(results_all['ap']):5.2f}")
        if args.save_pred:
            pred_all.append(torch.stack(pred, dim=0))
            true_all.append(torch.stack(true, dim=0))

    return results_all, pred_all, true_all


def main():
    args = parse_args()
    if args.cause:
        logger.print("Cause discovery")
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # load model
    set_seed(args.seed)
    model = load_model(args)
    pt = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(pt["state_dict"])
    model.eval()
    model.cuda()
    model.check_model(logger=logger)
    logger.print(f"Load checkpoint from {args.checkpoint_path}")
    logger.printt("Finished loading model.\n")

    # Start Testing
    logger.print("=== Starting Testing ===")
    logger.print(f"n_obs: {args.obs_size}, n_var: {args.num_vars}")
    logger.print(f"Ensemble: obs {args.en_obs}, vars {args.en_vars}")

    # save path
    path = f"/storage/janghyun/results/causal/sampling/{args.data_level[-1]}"
    os.makedirs(path, exist_ok=True)

    tag = ""
    if args.en_vars > 1:
        tag += f"_en{args.en_vars}"
    if args.tag != "":
        tag += f"_{args.tag}"
    print(f"Save results at {tag}")

    results_overall = defaultdict(list)
    for g in ["ecoli", "yeast_1k"]:
        logger.printt(f"\nGraph {g}")
        data_file = f"train_sergio_syn/{g}_add"
        args.data_file = [os.path.join(DATAPATH, data_file)]
        data = DataModule(args, verbose=False, only_pred=True, ensemble=True)

        results_all, pred_all, true_all = eval_model(args, model, data)

        logger.print("* Averaged score")
        for k in ["auroc", "ap"]:
            val = np.mean(results_all[k])
            results_overall[k].append(val)
            logger.print(f"{k:10s}: {val:.2f}")

        torch.save(results_all, f"{path}/results_{tag}_{g}.pt")
        if args.save_pred:
            torch.save(pred_all, f"{path}/pred_{tag}_{g}.pt")
            torch.save(true_all, f"{path}/true_{tag}_{g}.pt")

    for g in ["er", "sf", "sf_in", "sf_indirect"]:
        for e in [2, 4, 6]:
            logger.printt(f"\nGraph {g}, Edge {e}")
            data_file = f"train_sergio_syn/{g}_e{e}_add"
            args.data_file = [os.path.join(DATAPATH, data_file)]

            del data
            data = DataModule(args, verbose=False, only_pred=True, ensemble=True)

            results_all, pred_all, true_all = eval_model(args, model, data)

            logger.print("* Averaged score")
            for k in ["auroc", "ap"]:
                val = np.mean(results_all[k])
                results_overall[k].append(val)
                logger.print(f"{k:10s}: {val:.2f}")

            torch.save(results_all, f"{path}/results_{tag}_{g}_e{e}.pt")
            if args.save_pred:
                torch.save(pred_all, f"{path}/pred_{tag}_{g}_e{e}.pt")
                torch.save(true_all, f"{path}/true_{tag}_{g}_e{e}.pt")


if __name__ == "__main__":
    main()
