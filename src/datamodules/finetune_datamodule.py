import os
import random
import json
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

import lightning as L
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModelForImageClassification
from timm.data.transforms_factory import create_transform


class CustomImageFolder(ImageFolder):
    """
    Wrap torchvision.datasets.ImageFolder so that we can apply different transforms per
    class.
    We assume the classes are named "REAL" and "FAKE" (in that order), but we use label
    indices 0/1.
    """

    def __init__(self, root, transform_real=None, transform_fake=None):
        super().__init__(root)
        # We expect self.classes == ["REAL", "FAKE"] (alphabetical by default).
        # But in general, label 0 => "FAKE" if alphabetical; label 1 => "REAL".
        # To be robust, find index of "REAL" and "FAKE":
        self.class_to_idx = self.class_to_idx  # inherited
        self.idx_real = self.class_to_idx.get("REAL", None)
        self.idx_fake = self.class_to_idx.get("FAKE", None)
        self.transform_real = transform_real
        self.transform_fake = transform_fake

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        # Decide which transform to apply based on label:
        if label == self.idx_real and self.transform_real is not None:
            image = self.transform_real(image)
        elif label == self.idx_fake and self.transform_fake is not None:
            image = self.transform_fake(image)
        elif self.transform is not None:
            # fallback to transform if specified at parent level
            image = self.transform(image)
        return image, label

class DeepFakeFinetuningDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["model_name"]
        self.data_dir = cfg["data_dir"]
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["num_workers"]
        self.image_size_width = cfg["image_size_width"]
        self.image_size_height = cfg["image_size_height"]

        # Imbalance handling flags
        self.downsample = cfg.get("downsample", False)
        self.upsample = cfg.get("upsample", False)
        self.upsample_fraction = cfg.get("upsample_fraction", 1.0)
        self.augment_real = cfg.get("augment_real", False)

        # placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.class_counts = None

        self.idx_real = 0
        self.idx_fake = 1

    def _flatten_video_samples_to_frames(self, video_samples_info):
        flat_samples = []
        for _, frame_paths, label in video_samples_info:
            for frame_path in frame_paths:
                flat_samples.append((frame_path, label))
        return flat_samples

    def _prepare_transforms(self):
        # 1) Create a dummy model to extract mean/std/crop_pct from config
        tmp_model = AutoModelForImageClassification.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        cfg_model = tmp_model.config
        std = tmp_model.config.std
        mean = tmp_model.config.mean
        crop_mode = tmp_model.config.crop_mode
        crop_pct = tmp_model.config.crop_pct

        del tmp_model

        # 2) Define transforms
        # Base resize and center‐crop for validation/test
        self.val_transform = create_transform(
            input_size=(
                3,
                self.image_size_height,
                self.image_size_width,
            ),  # TIMM uses (C, H, W)
            is_training=False,
            mean=mean,
            std=std,
            crop_mode=crop_mode,
            crop_pct=crop_pct,
        )
        self.test_transform = self.val_transform

        # For training: define separate real/fake transforms
        # Fake: standard training augmentation
        train_base = create_transform(
            input_size=(self.image_size_height, self.image_size_width),
            is_training=True,
            mean=mean,
            std=std,
            crop_mode=crop_mode,
            crop_pct=crop_pct,
        )

        # If augment_real is True, apply the same augmentations + extra to REAL
        if self.augment_real:
            self.train_transform_real = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    train_base,
                ]
            )
        else:
            self.train_transform_real = train_base

        # Fake uses the standard train_base transform
        self.train_transform_fake = train_base

    def _get_video_ids_from_file(
        self, file_path_abs_or_rel_to_data_dir, data_dir_for_rel_path=None
    ):
        if not file_path_abs_or_rel_to_data_dir:
            return None

        actual_path = file_path_abs_or_rel_to_data_dir
        if not os.path.isabs(actual_path) and data_dir_for_rel_path:
            actual_path = os.path.join(data_dir_for_rel_path, actual_path)

        if not os.path.exists(actual_path):
            print(f"Warning: Split file not found: {actual_path}")
            return None
        with open(actual_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def setup(self, stage=None):

        self._prepare_transforms()

        # Build datasets
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        test_dir = os.path.join(self.data_dir, "test")

        if stage == "fit" or stage is None:
            # Use CustomImageFolder to route REAL vs FAKE to different transforms
            self.train_dataset = CustomImageFolder(
                train_dir,
                transform_real=self.train_transform_real,
                transform_fake=self.train_transform_fake,
            )
            self.val_dataset = CustomImageFolder(
                val_dir,
                transform_real=self.val_transform,
                transform_fake=self.val_transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = CustomImageFolder(
                test_dir,
                transform_real=self.test_transform,
                transform_fake=self.test_transform,
            )

        # Ensure we have both classes in train/val/test
        assert (
            self.train_dataset.idx_real is not None
            and self.train_dataset.idx_fake is not None
        ), "Dataset must contain both REAL and FAKE classes."
        assert (
            self.val_dataset.idx_real is not None
            and self.val_dataset.idx_fake is not None
        ), "Validation dataset must contain both REAL and FAKE classes."
        if self.test_dataset is not None:
            assert (
                self.test_dataset.idx_real is not None
                and self.test_dataset.idx_fake is not None
            ), "Test dataset must contain both REAL and FAKE classes."

        # 4) Compute class counts on TRAIN
        targets = [label for _, label in self.train_dataset.samples]
        class_counts = {}
        for lbl in targets:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        # Ensure order [count_real, count_fake]
        idx_real = self.train_dataset.idx_real
        idx_fake = self.train_dataset.idx_fake
        self.class_counts = {
            0: class_counts.get(0, 0),
            1: class_counts.get(1, 0),
        }
        # But map real/fake specifically:
        self.class_counts = {
            idx_real: class_counts.get(idx_real, 0),
            idx_fake: class_counts.get(idx_fake, 0),
        }

        # 5) Build WeightedRandomSampler for TRAIN (if needed)
        if self.downsample or self.upsample:
            count_real = self.class_counts[idx_real]
            count_fake = self.class_counts[idx_fake]

            if self.downsample:
                # we will sample 2 * count_real samples per epoch (equal REAL & FAKE)
                weights = []
                for _, label in self.train_dataset.samples:
                    if label == idx_real:
                        weights.append(1.0 / count_real)
                    else:
                        weights.append(1.0 / count_fake)
                num_samples = 2 * count_real

            # WARNING: This logic assumes that FAKE is the minority class.
            # If instead REAL is the minority class, this approach will incorrectly
            # upsample FAKE, potentially increasing class imbalance.
            elif self.upsample:
                # want to use fraction f of fakes; let fcount_fake = f * count_fake
                f = self.upsample_fraction
                fcount_fake = int(f * count_fake)
                # target: sample 2 * fcount_fake points (fcount_fake real + fcount_fake fake)
                weights = []
                for _, label in self.train_dataset.samples:
                    if label == idx_real:
                        weights.append(1.0 / count_real)
                    else:
                        weights.append(1.0 / count_fake)
                num_samples = 2 * fcount_fake

            self.train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True,
            )
        else:
            self.train_sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_class_counts(self):
        return self.class_counts
