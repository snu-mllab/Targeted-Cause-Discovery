import torch
from .model import CauseDiscovery


def load_model(args, load_ckpt=True, **kwargs):
    model = CauseDiscovery(args, **kwargs)
    if load_ckpt:
        _load_ckpt(model, args.checkpoint_path)

    print(f"Finished loading model from {args.checkpoint_path} (shift {args.shift}).")
    return model


def _load_ckpt(model, ckpt_path):
    pt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(pt["state_dict"])
    model.eval()
    model.cuda()


def set_checkpoint_path(args, inputs):
    n_obs, n_var = inputs[0].shape
    if n_obs < 200:
        raise AssertionError("#Observation should be at least 200")
    if n_var < 20:
        raise AssertionError("#Variable should be at least 20")

    if not args.en_obs:
        args.en_obs = 10

    if args.model_type == "continuous":
        if args.num_vars == 300:
            path = "./ckpt/continuous/model_best_step37000_epoch184_0.123.ckpt"
        elif args.num_vars == 200:
            path = "./ckpt/continuous/model_best_step38000_epoch189_0.125-v1.ckpt"
        elif args.num_vars == 100:
            path = "./ckpt/continuous/model_best_step38800_epoch193_0.127.ckpt"
        elif args.num_vars == 50:
            path = "./ckpt/continuous/model_best_step36600_epoch182_0.135.ckpt"
        elif args.num_vars == 30:
            path = "./ckpt/continuous/model_best_step32200_epoch160_0.141.ckpt"
        elif args.num_vars == 20:
            path = "./ckpt/continuous/model_best_step29400_epoch146_0.152.ckpt"
        en_vars = 5

    elif args.model_type == "discrete":
        # for umi count data
        path = "./ckpt/discrete/model_best_step18800_epoch93_0.159.ckpt"
        en_vars = 5

    if args.model_type == "impute":
        # for imputed data
        path = "./ckpt/impute/model_best_step21600_epoch107_0.155.ckpt"
        en_vars = 1

    if not args.en_vars:
        args.en_vars = en_vars

    args.checkpoint_path = path
