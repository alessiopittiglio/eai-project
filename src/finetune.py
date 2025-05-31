import argparse
import datetime
import logging
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# -------------------------------------------------------------------------
# 1) IMPORT HUGGING FACE
# -------------------------------------------------------------------------
from transformers import AutoModelForImageClassification, AutoConfig

# -------------------------------------------------------------------------
# 2) IMPORT YOUR EXISTING DATA MODULE (UNCHANGED)
# -------------------------------------------------------------------------
from datamodules import DeepfakeDataModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Make sure that GPU‐side matmul is using high precision (optional)
torch.set_float32_matmul_precision('high')


# -----------------------------------------------------------------------------
# 3) NEW LIGHTNINGMODULE: Wrap a HF image‐classification model for fine‐tuning
# -----------------------------------------------------------------------------
class HuggingFaceImageClassifier(L.LightningModule):
    """
    A simple LightningModule that loads a Hugging Face
    AutoModelForImageClassification (e.g. ViT, ResNet, etc.)
    with `num_labels = num_classes`, and fine‐tunes it on your dataset.
    
    Assumes each batch is a dict or tuple `(images, labels)` where:
      - images is a FloatTensor of shape (B, 3, H, W),
        already resized/normalized to what the HF model expects.
      - labels is a LongTensor of shape (B,) with integer class indices.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        use_scheduler: bool = False,
        scheduler_name: str = None,
        scheduler_params: dict = None,
        total_training_steps: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load HF config with the new number of labels
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_classes,
        )

        # 2) Load the pretrained model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

        # 3) (Optionally) keep track of weight decay, LR, scheduler, etc.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params or {}
        self.total_training_steps = total_training_steps

        # 4) Example metrics (you can add more, e.g. accuracy/Top‐1, Top‐5, etc.)
        self.train_acc = L.metrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = L.metrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = L.metrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, pixel_values, labels=None):
        """
        Forward pass: return HF model’s outputs.
        If `labels` is provided, HF model’s loss will be computed automatically.
        """
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        batch: tuple(images, labels)
          - images: Tensor[B,3,H,W]
          - labels: Tensor[B]
        """
        images, labels = batch
        # The HF model expects a kwarg `pixel_values=...`
        outputs = self.model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Log & track accuracy
        preds = torch.argmax(logits, dim=-1)
        self.train_acc.update(preds, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        self.val_acc.update(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(pixel_values=images, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        self.test_acc.update(preds, labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # We’ll use AdamW by default
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.use_scheduler:
            return optimizer

        # Example: a linear‐with‐warmup scheduler
        # You can adapt this to any HF scheduler you like.
        from transformers import get_linear_schedule_with_warmup

        # `self.total_training_steps` should be set to:
        #   num_epochs * (num_training_samples // batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.scheduler_params.get("warmup_steps", 0),
            num_training_steps=self.total_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",      # optional: if you want to tie to val loss
                "interval": "step",         # call scheduler.step() every optimizer step
            },
        }


# -----------------------------------------------------------------------------
# 4) MAIN TRAINING SCRIPT: almost identical to your original, but we swap in
#    `HuggingFaceImageClassifier` instead of `DeepfakeClassifier`.
# -----------------------------------------------------------------------------
def main(config):
    # 1) Timestamp & run_id
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = timestamp

    # 2) Seed for reproducibility
    L.seed_everything(config["seed"], workers=True)

    # 3) Create output directory
    output_path = Path(config["output_dir"]) / config["model"]["model_name"]
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output saved to: {output_path}")

    # 4) Instantiate your existing datamodule
    data_module = DeepfakeDataModule(
        data_dir=config["data"]["data_dir"],
        label_dirs=config["data"]["label_dirs"],
        load_sequences=config["data"]["load_sequences"],
        sequence_length=config["data"]["sequence_length"],
        sampling_stride=config["data"]["sampling_stride"],
        max_frames_per_video=config["data"]["max_frames_per_video"],
        max_videos_per_split=config["data"]["max_videos_per_split"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        transform_config=config["data"]["transform"],
    )
    data_module.setup("fit")

    # 5) Compute total training steps (for lr scheduler) if needed
    steps_per_epoch = len(data_module.train_dataset) // config["data"]["batch_size"]
    total_steps = config["trainer"]["max_epochs"] * steps_per_epoch

    # 6) Instantiate our HF‐based LightningModule
    model = HuggingFaceImageClassifier(
        model_name_or_path=config["model"]["model_name"],
        num_classes=config["model"]["num_classes"],
        learning_rate=config["model"].get("learning_rate", 5e-5),
        weight_decay=config["model"].get("weight_decay", 0.0),
        use_scheduler=config["model"].get("use_scheduler", False),
        scheduler_name=config["model"].get("scheduler_name", None),
        scheduler_params=config["model"].get("scheduler_params", {}),
        total_training_steps=total_steps,
    )

    # 7) Checkpointing (same as before)
    model_checkpoint = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename=f"{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    # 8) EarlyStopping (same as before)
    if config["trainer"]["callbacks"]["early_stopping"]:
        early_stopping = EarlyStopping(
            monitor="val/loss",
            patience=config["trainer"]["callbacks"]["early_stopping"]["patience"],
            mode="min",
        )
    else:
        raise ValueError("Early stopping is not enabled in the config file.")

    # 9) LearningRateMonitor (same)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 10) Logger (TensorBoard or WandB)
    pl_logger = None
    if config["logger"]["type"] == "tensorboard":
        pl_logger = TensorBoardLogger(save_dir=output_path)
    elif config["logger"]["type"] == "wandb":
        pl_logger = WandbLogger(
            project=config["project_name"],
            name=config["model"]["model_name"],
            save_dir=str(output_path),
        )

    # 11) Trainer (same signature)
    trainer = L.Trainer(
        accelerator=config["trainer"]["accelerator"],
        precision=config["trainer"]["precision"],
        devices=config["trainer"]["devices"],
        max_epochs=config["trainer"]["max_epochs"],
        callbacks=[model_checkpoint, early_stopping, lr_monitor],
        logger=pl_logger,
    )

    # 12) Fit
    logger.info("Starting fine‐tuning on Hugging Face model …")
    trainer.fit(model, datamodule=data_module)
    logger.info("Fine‐tuning completed.")

    # 13) Test using the best checkpoint
    best_ckpt = model_checkpoint.best_model_path
    logger.info(f"Testing with best checkpoint: {best_ckpt}")
    trainer.test(model=model, datamodule=data_module, ckpt_path=best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‐tune HF Model on Deepfake Data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load config YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config:")
    for k, v in sorted(config.items()):
        logger.info(f"  {k}: {v}")

    main(config)

