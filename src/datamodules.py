import lightning as L
import logging
from datasets import SingleFrameDataset, SequenceDataset
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Any
from utils import build_transforms

logger = logging.getLogger(__name__)

class DeepfakeDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_dir: str,
            label_dirs: Dict[str, int],
            # Dataset parameters
            load_sequences: bool = False,
            sequence_length: int = 16,
            sampling_stride: int = 1,
            max_frames_per_video: int = None,
            max_videos_per_split: int = None,
            # DataLoader parameters
            batch_size: int = 32,
            num_workers: int = 4,
            pin_memory: bool = True,
            transform_config: Optional[Dict[str, Any]] = None
        ):
        super().__init__()

        # Save arguments in hparams attribute
        # This allows to access them later in the code
        # and also to log them in the experiment tracker
        self.save_hyperparameters()
        
        self.train_tfm = None
        self.eval_tfm = None

        if self.hparams.load_sequences:
            logger.info(f"[DataModule] Configured to load SEQUENCES (T={self.hparams.sequence_length})")
            self.dataset_class = SequenceDataset
            self.dataset_specific_args = {
                "sequence_length": self.hparams.sequence_length,
                "sampling_stride": self.hparams.sampling_stride,
                "max_videos_per_split": self.hparams.max_videos_per_split
            }
        else:
            logger.info("[DataModule] Configured to load SINGLE FRAMES")
            self.dataset_class = SingleFrameDataset
            self.dataset_specific_args = {
                "max_frames_per_video": self.hparams.max_frames_per_video
            }

    def prepare_data(self):
        # Download and prepare the dataset
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        
        Args:
            stage (Optional[str], optional): Stage of the pipeline. Defaults to None.
                - "fit": training and validation
                - "test": testing
                - "predict": prediction
                - None: all stages
        """
        
        # Define the data preprocessing transforms
        transform_config = self.hparams.transform_config if self.hparams.transform_config is not None else {}
        
        # Build the transforms for training
        if not self.train_tfm:
            self.train_tfm = build_transforms(config=transform_config, augment=True)
        
        # Build the transforms for evaluation
        if not self.eval_tfm:
            self.eval_tfm = build_transforms(config=transform_config, augment=False)

        common_args = {
            "root_dir": self.hparams.data_dir,
            "label_dirs": self.hparams.label_dirs,
            **self.dataset_specific_args
        }

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(split='train', transform=self.train_tfm, **common_args)
            self.val_dataset = self.dataset_class(split='val', transform=self.eval_tfm, **common_args)

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(split='test', transform=self.eval_tfm, **common_args)

        if stage == "predict":
            self.predict_dataset = self.dataset_class(split='predict', transform=self.eval_tfm, **common_args)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """
        Helper for creating a DataLoader with common parameters.
        """
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=self.hparams.pin_memory,
        )
    
    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(self.predict_dataset, shuffle=False)
