import os
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser("")
parser.add_argument("--name", type=str, help="")
parser.add_argument("--idx", type=int, default=0, help="")
parser.add_argument("--size", type=int, default=10, help="")
parser.add_argument("--n_sample", type=int, default=100, help="")
args = parser.parse_args()

base_path = "/storage/janghyun/datasets/causal/train_sergio_syn"

path = os.path.join(base_path, args.name)
for i in range(args.size):
    idx = args.idx * args.size + i
    file_path = os.path.join(path, f"{idx}")

    print(i)
    for file in ["dropout.npy", "dropout_intv.npy"]:
        if "intv" in file:
            tag = "_intv"
        else:
            tag = ""

        data_orig = np.load(os.path.join(file_path, file))

        data_full = []
        for _ in range(5):
            data_sample = [np.random.poisson(data_orig) for _ in range(args.n_sample)]
            data_sample = np.stack(data_sample, axis=0).astype(np.float32)
            data_sample = np.sum(data_sample, axis=0) / args.n_sample
            data_full.append(data_sample)

        for k in [1]:
            data = np.sum(data_full[:k], axis=0) / k
            total = args.n_sample * k
            np.save(os.path.join(file_path, f"dropout_poisson{total}{tag}.npy"), data)

            err = np.mean(np.abs(data_orig - data))
            print(f"{err:.3f}", end=" ")

        zeros = np.mean(data_orig == 0)
        print(f"{zeros:.3f}", end=" ")
        print(flush=True)
