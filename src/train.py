import argparse
import datetime
import lightning as L
import logging
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from pathlib import Path
from datamodules import DeepfakeDataModule
from lightning_modules import DeepfakeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

def main(config):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = timestamp
    L.seed_everything(config['seed'], workers=True)

    output_path = Path(config['output_dir']) / config['model']['model_name']
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output saved to: {output_path}")
 
    data_module = DeepfakeDataModule(
        data_dir=config['data']['data_dir'],
        label_dirs=config['data']['label_dirs'],
        load_sequences=config['data']['load_sequences'],
        sequence_length=config['data']['sequence_length'],
        sampling_stride=config['data']['sampling_stride'],
        max_frames_per_video=config['data']['max_frames_per_video'],
        max_videos_per_split=config['data']['max_videos_per_split'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        transform_config=config['data']['transform'],
    )

    model = DeepfakeClassifier(
        model_name=config['model']['model_name'],
        learning_rate=config['model']['learning_rate'],
        optimizer_name=config['model']['optimizer_name'],
        use_scheduler=config['model']['use_scheduler'],
    )

    model_checkpoint = ModelCheckpoint(
        #dirpath=output_path / "checkpoints",
        filename=f"{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config['trainer']['callbacks']['early_stopping']['patience'],
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    pl_logger = None
    if config['logger']['type'] == "tensorboard":
        pl_logger = TensorBoardLogger(
            save_dir=output_path,
        )
    elif ['logger']['type'] == "wandb":
        pl_logger = WandbLogger(
            project=config['project_name'],
            #name=config['model]['model_name],
            #save_dir=output_path,
        )

    trainer = L.Trainer(
        accelerator=config['trainer']['accelerator'],
        precision=config['trainer']['precision'],
        devices=config['trainer']['devices'],
        max_epochs=config['trainer']['max_epochs'],
        callbacks=[model_checkpoint, early_stopping, lr_monitor],
        logger=pl_logger,
    )

    logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)
    logger.info("Training completed.")

    ckpt_path_for_test = model_checkpoint.best_model_path
    trainer.test(model=model, datamodule=data_module, ckpt_path=ckpt_path_for_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")

    parser.add_argument('--config', type=str, required=True,
                        help="Path to the YAML configuration file")
    
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("-- Training arguments --")
    for arg, value in sorted(config.items()):
        logger.info(f"{arg}: {value}")
    logger.info("--------------------------")

    main(config)
