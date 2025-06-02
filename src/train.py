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
    # Define timestamp and run ID
    # This is used for logging and saving the model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = timestamp
    
    # Set the random seed for reproducibility
    L.seed_everything(config['seed'], workers=True)

    # Set the output directory for saving the model and logs
    output_path = Path(config['output_dir']) / config['model']['model_name']
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output saved to: {output_path}")
    
    # Define the data module: 
    # this is used for loading the data
    # and splitting it into train, validation, and test sets
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
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    num_batches_per_epoch = len(train_dataloader)

    # Define the model
    model = DeepfakeClassifier(
        model_name=config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        model_params=config['model']['model_params'],
        criterion_name=config['model']['criterion_name'],
        criterion_params=config['model']['criterion_params'],
        optimizer_name=config['model']['optimizer_name'],
        optimizer_params=config['model']['optimizer_params'],
        use_scheduler=config['model']['use_scheduler'],
        scheduler_name=config['model']['scheduler_name'],
        scheduler_params=config['model']['scheduler_params'],
        accuracy_task=config['model']['accuracy_task'],
        accuracy_task_params=config['model']['accuracy_task_params'],
        # total_steps = num_epochs * num_batches_per_epoch
        scheduler_total_steps = config['trainer']['max_epochs'] * num_batches_per_epoch
        # num_warmup_steps = warmup_ratio * scheduler_total_steps
    )

    # Load the model weights if specified
    model_checkpoint = ModelCheckpoint(
        #dirpath=output_path / "checkpoints",
        filename=f"{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True
    )

    # Define the callbacks:
    # stopping the training early if the validation loss does not improve
    if config['trainer']['callbacks']['early_stopping']:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config['trainer']['callbacks']['early_stopping']['patience'],
            mode="min",
        )
    else:
        raise ValueError("Early stopping is not enabled in the config file.")

    # Define the learning rate monitor
    # this is used for logging the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Define the logger
    pl_logger = None
    if config['logger']['type'] == "tensorboard":
        pl_logger = TensorBoardLogger(
            save_dir=output_path,
        )
    elif config['logger']['type'] == "wandb":
        pl_logger = WandbLogger(
            project=config['project_name'],
            #name=config['model]['model_name],
            #save_dir=output_path,
        )

    # Define the trainer
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
