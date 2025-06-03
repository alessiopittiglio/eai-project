import argparse
import logging
import os
from collections import Counter  # <- added for per-batch class balance check

import yaml
import torch
from PIL import ImageFile

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.finetune_datamodule import DeepFakeFinetuningDataModule
from lightning_modules import DeepFakeFinetuningLightningModule

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def main(cfg):
    global logger
    # Set the random seed for reproducibility
    L.seed_everything(cfg["seed"], workers=True)

    # 1) Initialize DataModule
    dm = DeepFakeFinetuningDataModule(cfg)
    dm.setup()

    # 2) Get class counts from DataModule
    class_counts = dm.get_class_counts()
    logger.info(
        f"Train: REAL = {class_counts[dm.train_dataset.idx_real]}, "
        f"FAKE = {class_counts[dm.train_dataset.idx_fake]}"
    )

    # If you also want val/test:
    # Val
    val_labels = [lbl for _, lbl in dm.val_dataset.samples]
    val_counter = Counter(val_labels)
    logger.info(
        f"Val: REAL = {val_counter[dm.val_dataset.idx_real]}, "
        f"FAKE = {val_counter[dm.val_dataset.idx_fake]}"
    )

    # Test
    if dm.test_dataset is not None:
        test_labels = [lbl for _, lbl in dm.test_dataset.samples]
        test_counter = Counter(test_labels)
        logger.info(
            f"Test: REAL = {test_counter[dm.test_dataset.idx_real]}, "
            f"FAKE = {test_counter[dm.test_dataset.idx_fake]}"
        )

    # 2.5) Per-batch class balance check in the training loader
    logger.info("\nChecking class balance per batch in the training loader:")
    train_loader = dm.train_dataloader()
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        counts = Counter(labels.tolist())
        n_real = counts.get(dm.train_dataset.idx_real, 0)
        n_fake = counts.get(dm.train_dataset.idx_fake, 0)
        logger.info(f"Batch {batch_idx:03d} -> REAL: {n_real}, FAKE: {n_fake}")
        # Check only the first 10 batches for verification
        if batch_idx >= 9:
            break

    # 3) Initialize LightningModule
    model = DeepFakeFinetuningLightningModule(cfg, class_counts)

    # 4) Logger & Checkpoints
    pl_logger = TensorBoardLogger(
        save_dir=cfg["output_dir"],
        name=cfg["name_model"],
    )

    checkpoint_callback_best = ModelCheckpoint(
        monitor=cfg["monitor_metric"],
        mode=cfg["mode"],
        dirpath=pl_logger.log_dir,
        filename="best-{epoch:02d}-{" + cfg["monitor_metric"] + ":.4f}",
        save_top_k=1,
    )

    checkpoint_callback_last = ModelCheckpoint(
        dirpath=pl_logger.log_dir,
        filename="last",
        save_last=True,
    )

    # 5) Trainer
    early_stop_callback = EarlyStopping(
        monitor=cfg["monitor_metric"],
        mode=cfg["mode"],
        patience=cfg.get("early_stop_patience", 5),
        verbose=True,
    )

    # Define the learning rate monitor
    # this is used for logging the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        logger=pl_logger,
        callbacks=[
            checkpoint_callback_best,
            checkpoint_callback_last,
            early_stop_callback,
            lr_monitor,
        ],
        devices=cfg["devices"],
        precision=cfg["precision"],
        default_root_dir=tb_log_dir,
    )

    # 6) Train (option to resume from a previous checkpoint)
    resume_ckpt = cfg.get("resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, dm)

    # 7) Test
    test_results = trainer.test(model, datamodule=dm)
    with open(os.path.join(pl_logger.log_dir, "test_results.yaml"), "w") as f:
        yaml.safe_dump(test_results, f)

    # 8) Copy config to output directory
    with open(os.path.join(pl_logger.log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )

    with open(parser.parse_args().config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("-- Training arguments --")
    for arg, value in sorted(config.items()):
        logger.info(f"{arg}: {value}")
    logger.info("--------------------------")

    # Create output directories
    tb_log_dir = os.path.join(config["output_dir"], config["name_model"])
    os.makedirs(tb_log_dir, exist_ok=True)

    main(config)
