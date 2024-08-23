import os
import numpy as np
from collections import defaultdict
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from args import parse_args
from data import load_test_data
from model import load_model, set_checkpoint_path
from utils import set_seed
from inference import infer


def main(args):
    graph, inputs = load_test_data(args, index=args.env_idx)

    set_checkpoint_path(args, inputs)
    model = load_model(args, load_ckpt=True)

    print("\n=== Starting Testing ===", flush=True)
    print(f"n_obs: {args.obs_size} (en{args.en_obs}), n_var: {args.num_vars} (en{args.en_vars})",
          flush=True)
    path, file_tag = tagging(args)
    targets = set_targets(args, graph)
    pred = infer(args, model, inputs, targets)

    # Evaluate
    results = defaultdict(list)
    if graph is not None:
        auroc_fn = BinaryAUROC()
        ap_fn = BinaryAveragePrecision()

        if targets is not None:
            graph = graph[targets]
        for i in range(len(pred)):
            p, t = pred[i], graph[i]
            p, t = p[t != -100], t[t != -100]

            auroc = auroc_fn(p, t).item() * 100
            ap = ap_fn(p, t).item() * 100

            results["auroc"].append(auroc)
            results["ap"].append(ap)

    # save
    if args.save_pred:
        os.makedirs(path, exist_ok=True)
        if graph is not None:
            torch.save(results, f"{path}/results{file_tag}.pt")
            np.save(f"{path}/true{file_tag}.npy", graph.cpu().numpy())
        np.save(f"{path}/pred{file_tag}.npy", pred.cpu().numpy())
        print(f"Save results at {path}/pred{file_tag}.npy")


def set_targets(args, graph):
    # return indices of target variables
    targets = None
    if args.target_idx is not None:
        targets = np.arange(args.range) + args.range * args.target_idx
    else:
        if graph is not None:
            targets = torch.arange(len(graph))
            n_cause = graph.sum(1).cpu() + 100
            cond = n_cause >= 1  # only consider targets with at least one cause
            targets = targets[cond]

    if targets is not None:
        targets = targets.tolist()
        print(f"\n#Target nodes: {len(targets)}", flush=True)

    return targets


def tagging(args):
    name = args.data_file.split('/')[-1]
    path = os.path.join(args.save_path, f"{name}_{args.data_level}{args.tag}")

    file_tag = f"_var{args.num_vars}_enobs{args.en_obs}_envar{args.en_vars}"
    if args.anchor_size != 20:
        file_tag += f"_anchor{args.anchor_size}"
    if args.target_idx is not None:
        file_tag += f"_n{args.range}_{args.target_idx}"

    print(f"Save path: {path}")
    return path, file_tag


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    main(args)
