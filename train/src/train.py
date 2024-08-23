"""
Main training file. Call via train.sh
"""
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from args import parse_args
from data import DataModule
from model import load_model
from utils import Logger, get_suffix, set_seed

logger = Logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    args = parse_args()
    if args.cause:
        logger.print("Cause discovery")
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # save args (do not pickle object for readability)
    with open(args.args_file, "w+") as f:
        yaml.dump(args.__dict__, f)

    # setup
    set_seed(args.seed)
    model = load_model(args).cuda()
    model.check_model(logger=logger)
    logger.printt("Finished loading model.\n")

    # logger
    if args.debug:
        wandb_logger = None
    else:
        wandb_logger = None
        # wandb_logger = WandbLogger(project="causal", name=args.run_name)
        # wandb_logger.watch(model)  # gradients

    # data loaders
    data = DataModule(args)
    logger.printt("Finished loading raw data.\n")

    # train loop
    logger.print("=== Starting training ===")
    logger.print(f"n_obs: {args.obs_size}, n_var: {args.num_vars}")

    mode = "max"
    for keyword in ["loss"]:
        if keyword in args.metric:
            mode = "min"

    checkpoint_kwargs = {
        "save_top_k": 1,
        "monitor": args.metric,
        "mode": mode,
        "dirpath": args.save_path,
        "filename": get_suffix(args.metric),
        "auto_insert_metric_name": False,
        "save_last": not args.not_save_last,
        "verbose": True,
    }
    # checkpoint_path is a PTH to resume training
    if os.path.exists(args.checkpoint_path):
        checkpoint_kwargs["dirpath"] = args.checkpoint_path
    cb_checkpoint = ModelCheckpoint(**checkpoint_kwargs)
    logger.print(f"Model save path: {checkpoint_kwargs['dirpath']}")

    cb_earlystop = EarlyStopping(
        monitor=args.metric,
        patience=args.patience,
        mode=mode,
    )
    cb_lr = LearningRateMonitor(logging_interval="step")
    callbacks = [
        RichProgressBar(),
        cb_checkpoint,
        # cb_earlystop,
        # cb_lr
    ]
    if args.patience > 0:
        callbacks.append(cb_earlystop)

    if args.no_tqdm:
        callbacks[0].disable()

    device_ids = [args.gpu + i for i in range(args.num_gpu)]

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "min_epochs": 0,
        "accumulate_grad_batches": args.accumulate_batches,
        "gradient_clip_val": 1.,
        # evaluate more frequently
        "limit_train_batches": 20 * args.accumulate_batches,
        "limit_val_batches": 20 * args.accumulate_batches,
        # logging and saving
        "callbacks": callbacks,
        "log_every_n_steps": args.log_frequency,
        "fast_dev_run": args.debug,
        "logger": wandb_logger,
        # GPU utilization
        "devices": device_ids,
        "accelerator": "gpu",
        # "strategy": "ddp"
    }

    trainer = pl.Trainer(**trainer_kwargs)
    logger.printt("Initialized trainer.")

    # if applicable, restore full training
    fit_kwargs = {}
    if os.path.exists(args.checkpoint_path):
        fit_kwargs["ckpt_path"] = args.checkpoint_path

    model.train()
    trainer.fit(model, data, **fit_kwargs)
    logger.printt("All done.")


if __name__ == "__main__":
    main()
