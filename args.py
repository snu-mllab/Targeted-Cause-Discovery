import os
import argparse

DATAPATH = "/storage/janghyun/datasets/causal"
SAVEPATH = "/storage/janghyun/results/causal/inference"


def get_parser():
    parser = argparse.ArgumentParser("")

    # ======== data ========
    parser.add_argument("--data_file",
                        type=str,
                        default="train_sergio_syn/ecoli_add",
                        help="dataset folder name")
    parser.add_argument("--fidelity",
                        type=str,
                        default="mid",
                        choices=["low", "mid", "high"],
                        help="simulator fidelity")
    parser.add_argument("--data_level", type=str, default="dropout")

    parser.add_argument("--env_idx", type=int, default=0, help="dataset environment index")
    parser.add_argument("--target_idx", type=int, default=None, help="target index")
    parser.add_argument("--range", type=int, default=1, help="range of targets to test")

    # input configuration
    parser.add_argument("--num_vars", type=int, default=200, help="number of variables per input")
    parser.add_argument("--obs_size",
                        type=int,
                        default=200,
                        help="number of observations per input")
    parser.add_argument("--obs_ratio", type=float, default=0.05, help="ratio of observational data")

    # data preprocessing
    parser.add_argument("--log", type=str2bool, default=True, help="use log normalization")
    parser.add_argument("--shift",
                        type=float,
                        default=None,
                        help="shift value after log normalization")
    parser.add_argument("--dtype", type=str, default="float32", help="data type")

    # ensemble configuration
    parser.add_argument("--en_obs", type=int, default=None, help="observation ensemble size")
    parser.add_argument("--en_vars", type=int, default=None, help="variable ensemble size")
    parser.add_argument("--anchor_size",
                        type=int,
                        default=20,
                        help="size of a fixed variable set during ensembled local-inference")

    # ======== model =======
    parser.add_argument("--model_type",
                        type=str,
                        default="discrete",
                        choices=["continuous", "discrete", "impute"])
    parser.add_argument("-l", "--transformer_num_layers", type=int, default=10)
    parser.add_argument("--n_heads", type=int, default=16, help="attention heads")
    parser.add_argument("--embed_dim", type=int, default=16, help="embedding size")
    parser.add_argument("--ffn_embed_dim", type=int, default=96, help="Transformer ffn size")
    parser.add_argument("--dropout", type=float, default=0.)

    # ======== misc =======
    parser.add_argument("--save_pred",
                        type=str2bool,
                        default=True,
                        help="Save predictions on test set")
    parser.add_argument("--print_freq", type=int, default=20, help="time stamp period")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--tag", type=str, default="", help="tag for saving results")

    return parser


def parse_args():
    args = get_parser().parse_args()
    if args.data_file != "human":
        args.data_file = os.path.join(DATAPATH, args.data_file)
    else:
        args.model_type = "discrete"
        args.en_vars = 1  # setting used in our paper.

    args.data_path = DATAPATH
    args.save_path = SAVEPATH

    if args.model_type == "continuous":
        args.shift = 12
    elif args.model_type == "discrete":
        args.shift = 10
    elif args.model_type == "impute":
        args.shift = 10

    if args.tag:
        args.tag = "_" + args.tag
    return args


def override_args(args, config, skip=set()):
    for k, v in config.items():
        if k in skip:
            continue
        if type(v) is dict:
            override_args(args, v)
        else:
            args.__dict__[k] = v
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
