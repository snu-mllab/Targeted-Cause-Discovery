import os
import sys
import yaml
import argparse

from utils import printt

DATAPATH = "/storage/janghyun/datasets/causal"
SAVEPATH = "/storage/janghyun/results/causal"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser("")
    # configuration
    parser.add_argument("--config_file", type=str, default=None, help="YAML file")
    parser.add_argument("--args_file",
                        type=str,
                        default="args.yaml",
                        help="Dump arguments for reproducibility")
    parser.add_argument("--results_file",
                        type=str,
                        default="results.yaml",
                        help="Save outputs here")

    # ======== data ========
    # include path to pdb protein files, splits
    parser.add_argument("--cause", action="store_true", help="cause discovery")
    parser.add_argument("--data_file",
                        type=str,
                        nargs='+',
                        default=["test"],
                        help="Data file folder")
    parser.add_argument("--data_level", type=str, nargs='+', default=["dropout"])
    parser.add_argument("--n_env", type=int, default=50)

    parser.add_argument("-n",
                        "--num_vars",
                        type=int,
                        default=200,
                        help="number of variables per input")
    parser.add_argument("--obs_size",
                        type=int,
                        default=200,
                        help="number of observations per input")
    parser.add_argument("--obs_ratio", type=float, default=0.05)
    parser.add_argument("--only_obs", action="store_true")

    # ensemble
    parser.add_argument("--en_obs", type=int, default=10, help="observation ensemble")
    parser.add_argument("--en_vars", type=int, default=5, help="variable ensemble")
    parser.add_argument("--anchor_size", type=int, default=20)

    # data loading
    parser.add_argument("--log", type=str2bool, default=True, help="")
    parser.add_argument("--shift", type=float, default=12)

    parser.add_argument("--num_workers", type=int, default=10, help="data loader workers")
    parser.add_argument("--dtype", type=str, default="float32")

    # ======== model =======
    parser.add_argument("--model", type=str, default="cause", help="aggregator")
    parser.add_argument("-l",
                        "--transformer_num_layers",
                        type=int,
                        default=10,
                        help="number of 2d transformer blocks")
    parser.add_argument("--n_heads", type=int, default=16, help="attention heads")
    parser.add_argument("--embed_dim", type=int, default=16, help="embed size")
    parser.add_argument("--ffn_embed_dim", type=int, default=96, help="Transformer ffn size")
    parser.add_argument("--output_dim", type=int, default=-1, help="output size")

    # ====== training ======
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs to train")
    parser.add_argument("--patience",
                        type=int,
                        default=-1,
                        help="Lack of validation improvement for [n] epochs")
    parser.add_argument("--metric", type=str, default="Val/loss")

    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--num_gpu", type=int, default=1, help="number of GPUs")
    parser.add_argument("--seed", type=int, default=0, help="Initial seed")

    # ==== optimization ====
    parser.add_argument("--batch_size", type=int, default=32, help="number of graphs per batch")
    parser.add_argument("--accumulate_batches", type=int, default=1, help="accumulate gradient")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="cosine", help="cosine,constant")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization weight")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout probability")

    # ==== logging ====
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_frequency",
                        type=int,
                        default=10,
                        help="log to wandb every [n] batches")
    parser.add_argument("--save_pred", action="store_true", help="Save predictions on test set")
    parser.add_argument("--no_tqdm", action="store_true", help="dispatcher mode")
    parser.add_argument("--not_save_last",
                        action="store_true",
                        help="Set flag true to load smaller dataset")

    # (optional)
    parser.add_argument("-c",
                        "--checkpoint_path",
                        type=str,
                        default="",
                        help="Checkpoint for entire model for test/finetune")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Set flag true to load smaller dataset")
    parser.add_argument("--tag", type=str, default="")

    return parser


def parse_args():
    args = get_parser().parse_args()
    process_args(args)
    return args


def process_args(args):
    args.data_file = [os.path.join(DATAPATH, p) for p in args.data_file]

    if args.patience == -1:
        args.patience = args.epochs // 10

    # save path
    SAVEPATH_ = SAVEPATH
    if args.run_name == "":
        args.run_name = "_".join([file.split("/")[-1][5:] for file in args.data_file])
    elif args.run_name[:5] == "slurm":
        SAVEPATH_ = "./results"

    args.save_path = os.path.join(SAVEPATH_, args.run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # load configuration = override specified values
    if args.model == "aggregator":
        args.config_file = "/storage/janghyun/results/causal/gies_synthetic/args.yaml"

    if args.config_file is not None:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        k_to_skip = {"save_pred"}
        override_args(args, config, skip=k_to_skip)

    # finally load all saved parameters
    if args.model == "aggregator":
        args.checkpoint_path = "gies_synthetic/model_best_epoch=535_auprc=0.849.ckpt"

    if len(args.checkpoint_path) > 0 and args.checkpoint_path[:2] != "./":
        args.checkpoint_path = os.path.join(SAVEPATH_, args.checkpoint_path)
        if not os.path.exists(args.checkpoint_path):
            printt("invalid checkpoint_path", args.checkpoint_path)
            sys.exit(0)

        with open(os.path.join(os.path.dirname(args.checkpoint_path), args.args_file)) as f:
            config = yaml.safe_load(f)

        # do not overwrite certain args
        k_to_skip = {"gpu", "debug", "num_workers", "save_pred"}
        for k in config:
            if "file" in k:
                k_to_skip.add(k)
            if "path" in k:
                k_to_skip.add(k)
            if "batch" in k:
                k_to_skip.add(k)
        override_args(args, config, skip=k_to_skip)

    if args.output_dim == -1:
        args.output_dim = args.embed_dim

    # prepend output root
    args.args_file = os.path.join(args.save_path, args.args_file)

    args.results_file = f"results"
    if args.cause:
        args.results_file += "_cause"

    args.results_file += f"_obs{args.obs_size}_node{args.num_vars}"
    args.results_file += f"_l{args.transformer_num_layers}_embed{args.embed_dim}_ffn{args.ffn_embed_dim}_head{args.n_heads}"
    args.results_file += f"_bs{args.batch_size}_lr{args.lr}_dropout{args.dropout}"

    args.results_file += f"_oratio{args.obs_ratio}"
    if args.shift is not None:
        args.results_file += f"_shift{args.shift}"

    if args.only_obs:
        args.results_file += f"_onlyobs"

    if args.n_env != 100:
        args.results_file += f"_env{args.n_env}"
    if args.dtype != "float32":
        args.results_file += f"_{args.dtype}"

    args.results_file += f"_epoch{args.epochs}_{args.seed}"
    args.results_file = os.path.join(args.save_path, args.results_file + ".yaml")

    print(args.data_level, flush=True)
    print(args.results_file, flush=True)


def override_args(args, config, skip=set()):
    """
        Recursively copy over config to args
    """
    for k, v in config.items():
        if k in skip:
            continue
        if type(v) is dict:
            override_args(args, v)
        else:
            args.__dict__[k] = v
    return args
