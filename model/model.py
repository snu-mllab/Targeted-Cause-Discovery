# Codes are adapted from: https://github.com/rmwu/sea (Author: Menghua Wu)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from .axial_cause import AxialTransformer, TopLayer_Perm


class CauseDiscovery(pl.LightningModule):
    """ input : data, intervention matrix (n_obs x n_var)
        output: cause scores among variables (n_var x n_var)
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_vars = args.num_vars

        # Transformer on sequence of predicted graphs
        self.encoder = AxialTransformer(args)

        self.top_layer = TopLayer_Perm(args)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = BinaryAUROC()
        self.ap = BinaryAveragePrecision()

        self.save_hyperparameters()

    def forward(self, batch):
        """
        Used on predict_dataloader
        """

        with torch.no_grad():
            x = self.encoder(batch)
            logit = self.top_layer(x)

        results = self.compute_losses(logit, batch["label"])
        results["loss"] = results["loss"].item()
        results.update(self.compute_acc(logit, batch["label"]))

        return results

    def compute_losses(self, logit, label, ignore_index=-100):
        losses = {}
        logit = logit.reshape(-1)
        label = label.reshape(-1).to(logit.dtype)
        index = label != ignore_index

        losses["loss"] = self.loss_fn(logit[index], label[index])
        return losses

    def compute_acc(self, logit, label, ignore_index=-100):
        logit = logit.cpu()
        label = label.cpu()

        auroc, ap, acc = [], [], []
        if self.args.save_pred:
            pred_list, true_list = [], []

        with torch.no_grad():
            for i in range(len(logit)):
                if self.args.save_pred:
                    pred_list.append(logit[i])
                    true_list.append(label[i])

                p = logit[i].reshape(-1)
                t = label[i].reshape(-1)

                index = t != ignore_index
                p = p[index]
                t = t[index]

                auroc.append(self.auroc(p, t).item() * 100)
                ap.append(self.ap(p, t).item() * 100)

                p = p > 0.
                t = t.to(p.dtype)
                acc.append((p == t).float().mean().item() * 100)

        outputs = {}
        outputs["auroc"] = auroc
        outputs["ap"] = ap
        outputs["acc"] = acc
        if self.args.save_pred:
            outputs["pred"] = pred_list
            outputs["true"] = true_list

        return outputs

    def training_step(self, batch, batch_idx):
        x = self.encoder(batch)
        logit = self.top_layer(x)
        losses = self.compute_losses(logit, batch["label"])

        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Train/{k}", v.item(), batch_size=len(logit), sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x = self.encoder(batch)
        logit = self.top_layer(x)
        losses = self.compute_losses(logit, batch["label"])

        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Val/{k}", v.item(), batch_size=len(logit), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        param_groups = get_params_groups(self, self.args)
        self.optimizer = AdamW(param_groups)

        if self.args.scheduler == "cosine":
            # scheduler makes everything worse =__=
            scheduler = CosineAnnealingLR(self.optimizer, self.args.epochs)
            return [self.optimizer], [scheduler]
        else:
            return [self.optimizer]

    def check_model(self, logger=None):
        mem = torch.cuda.memory_allocated() / 10**6
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if logger is not None:
            logger.print(f"\nCheck model: {self.dtype}, {self.device} (current Mem {mem:.0f} MB)")
            logger.print(f"Number of trainable parameters: {pytorch_total_params:,}")
        else:
            print(f"\nCheck model: {self.dtype}, {self.device} (current Mem {mem:.0f} MB)")
            print(f"Number of trainable parameters: {pytorch_total_params:,}")


def get_params_groups(model, args):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    groups = [
        {
            "params": regularized,
            "weight_decay": args.weight_decay,
            "lr": args.lr
        },
        {
            "params": not_regularized,
            "weight_decay": 0.,
            "lr": args.lr
        },
    ]
    return groups
