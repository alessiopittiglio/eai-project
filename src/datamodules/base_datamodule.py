import logging
from typing import Dict, Optional, Any
from collections import defaultdict, Counter

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from datasets import SingleFrameDataset, SequenceDataset
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
            transform_config: Optional[Dict[str, Any]] = None,
            use_weighted_sampler: bool = True,
            seed: int = 42
        ):
        super().__init__()

        # Save arguments in hparams attribute
        # This allows to access them later in the code
        # and also to log them in the experiment tracker
        self.save_hyperparameters()

        self._set_dataset_class()
        
        self.train_tfm = None
        self.eval_tfm = None

    def _set_dataset_class(self):
        """Configure dataset type and related parameters."""
        if self.hparams.load_sequences:
            logger.info(
                f"[DataModule] Loading SEQUENCES (T={self.hparams.sequence_length})"
            )
            self.dataset_class = SequenceDataset
            self.dataset_args = {
                "sequence_length": self.hparams.sequence_length,
                "sampling_stride": self.hparams.sampling_stride,
                "max_videos_per_split": self.hparams.max_videos_per_split,
            }
        else:
            logger.info("[DataModule] Loading SINGLE FRAMES")
            self.dataset_class = SingleFrameDataset
            self.dataset_args = {
                "max_frames_per_video": self.hparams.max_frames_per_video,
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
            **self.dataset_args
        }

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(split='train', transform=self.train_tfm, **common_args)
            self.val_dataset = self.dataset_class(split='val', transform=self.eval_tfm, **common_args)

            if self.hparams.use_weighted_sampler and self.train_dataset and len(self.train_dataset) > 0:
                self.train_sampler = self._build_weighted_sampler(self.train_dataset.targets)

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(split='test', transform=self.eval_tfm, **common_args)

        if stage == "predict":
            self.predict_dataset = self.dataset_class(split='predict', transform=self.eval_tfm, **common_args)

        self._log_split_counts()
    
    def _build_weighted_sampler(self, targets: list) -> Optional[WeightedRandomSampler]:
        if not targets:
            return None

        class_counts = Counter(targets)
        logger.info(f"[DataModule] Train class counts for sampler: {dict(class_counts)}")

        if len(class_counts) < 2:
            return None

        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[t] for t in targets]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )
            
    def _create_dataloader(self, dataset: Dataset, shuffle: bool, sampler: Optional[WeightedRandomSampler] = None) -> DataLoader:
        """
        Helper for creating a DataLoader with common parameters.
        """

        actual_shuffle = shuffle if sampler is None else False

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=actual_shuffle,
            sampler=sampler,
            pin_memory=self.hparams.pin_memory,
        )
    
    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True, sampler=self.train_sampler)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(self.predict_dataset, shuffle=False)
    
    def _log_split_counts(self):
        """Log number of samples per class for each of train/val/test."""
        # invert label_dirs so we can map int→name
        inv_map = {v:k for k, v in self.hparams.label_dirs.items()}

        for split in ("train", "val", "test"):
            try:
                ds = getattr(self, f"{split}_dataset")
                counts = defaultdict(int)
                for _, lbl in ds.samples:
                    counts[inv_map[lbl]] += 1
                # e.g. "[DataModule] train: real=123, fake=123"
                counts_str = ", ".join(f"{cls}={cnt}" for cls, cnt in counts.items())
                logger.info(f"[DeepfakeDataModule] {split}: {counts_str}")
            except AttributeError:
                logger.warning(f"[DeepfakeDataModule] {split} dataset not found. Skipping logging.")
