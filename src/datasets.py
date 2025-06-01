import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import defaultdict


class BaseFramesDataset(Dataset):
    """
    Abstract base class for datasets built on frames pre-extracted from videos.

    Now with automatic undersampling to balance all labels
    down to the count of the least-represented label.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        label_dirs: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.label_dirs = label_dirs
        # Default to ToTensor if no transform provided
        self.transform = transform or T.Compose([T.ToTensor()])
        self.split_dir = self.root_dir / self.split

        # load, then balance
        self.samples: List[Tuple] = self._load_samples()
        # self._balance_samples()
        self._log_dataset_info()

    def _load_samples(self) -> List[Tuple]:
        """
        Subclasses must implement loading of dataset samples.
        Returns:
            List of tuples, typically (data_path, label_int).
        """
        raise NotImplementedError

    def _balance_samples(self) -> None:
        """
        Undersample every class to the size of the smallest class.
        """
        logger.info(f"[Dataset: {self.__class__.__name__}] Balancing samples in {self.split_dir}")
        # group samples by label
        grouped: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
        for sample in self.samples:
            _, label = sample
            grouped[label].append(sample)

        # find minimum class size
        min_count = min(len(samples) for samples in grouped.values())

        # undersample each group
        balanced = []
        for label, samples in grouped.items():
            if len(samples) > min_count:
                balanced.extend(random.sample(samples, min_count))
            else:
                balanced.extend(samples)

        random.shuffle(balanced)
        self.samples = balanced

    def _log_dataset_info(self) -> None:
        if not self.samples:
            logger.warning(f"[Dataset: {self.__class__.__name__}] No samples found in {self.split_dir}")
        else:
            # report counts per class
            counts = defaultdict(int)
            for _, lbl in self.samples:
                counts[lbl] += 1
            counts_str = ", ".join(f"{lbl}={cnt}" for lbl, cnt in counts.items())
            logger.info(f"[Dataset: {self.__class__.__name__}] Loaded {len(self.samples)} samples ({counts_str}) from {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        """
        Safely load an image and convert to RGB.
        Returns a black 224x224 placeholder on failure.
        """
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return Image.new('RGB', (224, 224))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split='{self.split}', samples={len(self)})"


class SingleFrameDataset(BaseFramesDataset):
    """
    PyTorch Dataset for loading single frames pre-extracted from videos.

    Directory structure:
        <root_dir>/<split>/<label_dir>/<video_id>/<frame_id>.(png|jpg|jpeg)

    Returns:
        Tuple[Tensor[C,H,W], int] per sample.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        label_dirs: Dict[str, int],
        transform: Optional[Callable] = None,
        max_frames_per_video: Optional[int] = None,
    ):
        self.max_frames_per_video = max_frames_per_video
        super().__init__(root_dir, split, label_dirs, transform)

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        logger.info(f"[Dataset: {self.__class__.__name__}] Scanning frames in {self.split_dir}")

        for label_name, label_int in self.label_dirs.items():
            label_dir = self.split_dir / label_name
            if not label_dir.is_dir():
                logger.warning(f"Label directory {label_dir} not found, skipping")
                continue

            for video_dir in sorted(label_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                frame_paths = sorted(
                    p for p in video_dir.iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
                )
                if not frame_paths:
                    continue

                # Limit frames per video if requested
                if self.max_frames_per_video and len(frame_paths) > self.max_frames_per_video:
                    frame_paths = frame_paths[: self.max_frames_per_video]

                for fp in frame_paths:
                    samples.append((fp, label_int))

        return samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = self._load_image(path)
        tensor = self.transform(img)
        return tensor, label


class SequenceDataset(BaseFramesDataset):
    """
    PyTorch Dataset for loading sequences of frames (clips) pre-extracted from videos.

    Directory structure:
        <root_dir>/<split>/<label_dir>/<video_id>/<frame_id>.(png|jpg|jpeg)

    Returns:
        Tuple[Tensor[C,T,H,W], int] per sample.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        label_dirs: Dict[str, int],
        sequence_length: int,
        sampling_stride: int = 1,
        transform: Optional[Callable] = None,
        max_videos_per_split: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.sampling_stride = sampling_stride
        self.max_videos = max_videos_per_split
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        super().__init__(root_dir, split, label_dirs, transform)
        logger.info(f"[Dataset: {self.__class__.__name__}] sequence_length={sequence_length}, stride={sampling_stride}")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        logger.info(f"[Dataset: {self.__class__.__name__}] Scanning video directories in {self.split_dir}")

        for label_name, label_int in self.label_dirs.items():
            label_dir = self.split_dir / label_name
            if not label_dir.is_dir():
                logger.warning(f"Label directory {label_dir} not found, skipping")
                continue

            for video_dir in sorted(label_dir.iterdir()):
                if video_dir.is_dir():
                    samples.append((video_dir, label_int))

        if self.max_videos and len(samples) > self.max_videos:
            logger.info(f"Limiting to first {self.max_videos} videos out of {len(samples)}")
            samples = samples[: self.max_videos]

        return samples

    def _sample_frame_indices(self, num_frames: int) -> List[int]:
        """
        Sample frame indices given available frame count.
        If enough frames exist, pick a contiguous clip with stride; else sample with replacement.
        """
        if num_frames <= 0:
            return []

        needed = (self.sequence_length - 1) * self.sampling_stride + 1
        if num_frames >= needed:
            max_start = num_frames - needed
            start = random.randint(0, max_start)
            return [start + i * self.sampling_stride for i in range(self.sequence_length)]
        else:
            idxs = np.random.choice(num_frames, self.sequence_length, replace=True)
            return sorted(int(i) for i in idxs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_dir, label = self.samples[idx]
        frame_paths = sorted(
            p for p in video_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        )
        num_frames = len(frame_paths)

        indices = self._sample_frame_indices(num_frames)
        if not indices:
            logger.error(f"No frames to sample for {video_dir}, returning dummy tensor")
            dummy = torch.zeros((3, self.sequence_length, 224, 224))
            return dummy, label

        frames = []
        for i in indices:
            img = self._load_image(frame_paths[i])
            frames.append(self.transform(img))  # [C,H,W]

        clip = torch.stack(frames, dim=1)  # [C, T, H, W]
        return clip, label
