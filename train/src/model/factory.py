import torch
from .model import CauseDiscovery


def load_model(args, load_ckpt=False, **kwargs):
    """
        Model factory
    """
    model = CauseDiscovery(args, **kwargs)
    if load_ckpt:
        _load_ckpt(model, args.checkpoint_path)

    return model


def _load_ckpt(model, ckpt_path):
    pt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(pt["state_dict"])
    model.eval()
    model.cuda()
