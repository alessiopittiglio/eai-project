import argparse
import lightning as L
import logging
import yaml
import torch
from pathlib import Path
from datamodules.base_datamodule import DeepfakeDataModule
from lightning_modules import DeepfakeClassifier
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def main(config):
    L.seed_everything(config["seed"], workers=True)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

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

    eval_output_path = Path(config["output_dir"]) / f"eval_{checkpoint_path.stem}"
    eval_output_path.mkdir(parents=True, exist_ok=True)

    model = DeepfakeClassifier.load_from_checkpoint(args.checkpoint_path)

    model_checkpoint = ModelCheckpoint(
        # dirpath=output_path / "checkpoints",
        filename=f"{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    if config["trainer"]["callbacks"]["early_stopping"]:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config["trainer"]["callbacks"]["early_stopping"]["patience"],
            mode="min",
        )
    else:
        raise ValueError("Early stopping is not enabled in the config file.")

    trainer = L.Trainer(
        accelerator=config["trainer"]["accelerator"],
        precision=config["trainer"]["precision"],
        devices=config["trainer"]["devices"],
        max_epochs=config["trainer"]["max_epochs"],
        callbacks=[model_checkpoint, early_stopping],
    )

    logger.info(f"Starting test on checkpoint: {checkpoint_path}")
    test_results = trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=args.checkpoint_path,
        verbose=True,
    )
    logger.info("Test result:")
    for key, value in test_results[0].items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )

    args = parser.parse_args()

    with open(parser.parse_args().config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("-- Evaluation configuration parameters --")
    for arg, value in sorted(config.items()):
        logger.info(f"{arg}: {value}")
    logger.info("-----------------------------------------")

    main(config)
