import logging
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing import Dict, Callable, Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The first approach was "video on-the-fly". Conceptually, it's correct, 
# but it is not efficient. Every time we access to an item, we have to 
# load the video from disk and decode it.

class BaseFramesDataset(Dataset):
    """
    Abstract base class for datasets based on frames pre-extracted from videos.
    """
    def __init__(
            self, 
            root_dir: str,
            split: str,
            label_dirs: Dict[str, int],
            transform: Optional[Callable] = None
        ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.label_map = label_dirs
        self.transform = transform
        self.split_dir = self.root_dir / self.split

        self.samples = self._load_samples()
        self._log_dataset_info()

    def _load_samples(self) -> List:
        raise NotImplementedError 

    def _log_dataset_info(self):
        if not self.samples:
            logger.warning(f"[Dataset: {self.__class__.__name__}] No samples found in {self.split_dir}.")
        else:
            logger.info(f"[Dataset: {self.__class__.__name__}] Found {len(self.samples)} samples in {self.split_dir}.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, path: Path) -> Image.Image:
        """
        Load an image from the given path.
        """
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            logger.error(f"[Dataset: {self.__class__.__name__}] Error loading image {path}: {e}")
            raise e
        return img

class SingleFrameDataset(BaseFramesDataset):
    """
    Dataset PyTorch for loading SINGLE video frames pre-extracted.

    Optimized for 2D models (e.g. ResNet18 or ViT).

    Directory structure:
    `<root_dir>/<split>/<label_dir>/<video_id>/<frame_id>.(png|jpg)`

    Output __getitem__: (tensor[C, H, W], label_int)
    """
    def __init__(
        self, 
        root_dir: str,
        split: str,
        label_dirs: Dict[str, int],
        transform: Optional[Callable] = None, 
        max_frames_per_video: Optional[int] = None
    ):
        self.max_frames_per_video = max_frames_per_video
        super().__init__(root_dir, split, label_dirs, transform)

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Load pairs of (frame_path, label_int).
        """
        samples = []
        logger.info(f"[Dataset: {self.__class__.__name__}] Scanning single frames...")
        for label_dir_name, label_int in self.label_map.items():
            label_dir_path = self.split_dir / label_dir_name
            if not label_dir_path.is_dir(): continue

            for video_dir in label_dir_path.iterdir():
                if not video_dir.is_dir(): continue
                frame_files = sorted([p for p in video_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                if not frame_files: continue

                if self.max_frames_per_video is not None and len(frame_files) > self.max_frames_per_video:
                    # Easy way: take the first N frames
                    sampled_frames = frame_files[:self.max_frames_per_video]
                else:
                    sampled_frames = frame_files

                for frame_path in sampled_frames:
                    samples.append((frame_path, label_int))

        return samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_path, label = self.samples[idx]

        img = self._load_image(frame_path)
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        return img_tensor, label

class SequenceDataset(BaseFramesDataset):
    """
    Dataset PyTorch for loading SEQUENCE video frames pre-extracted.

    Optimized for 3D models (e.g. Xception3D)

    Directory structure:
    `<root_dir>/<split>/<label_dir>/<video_id>/<frame_id>.(png|jpg)`

    Output __getitem__: (tensor[C, T, H, W], label_int)
    """
    def __init__(
            self, 
            root_dir: str,
            split: str,
            label_dirs: Dict[str, int],
            sequence_length: int,
            sampling_stride: int = 1,
            transform: Optional[Callable] = None,
            max_videos_per_split: Optional[int] = None
        ):
        self.sequence_length = sequence_length
        self.sampling_stride = sampling_stride
        self.max_videos = max_videos_per_split # Renamed for clarity
        super().__init__(root_dir, split, label_dirs, transform)
        logger.info(f"[Dataset: {self.__class__.__name__}] T={self.sequence_length}, Stride={self.sampling_stride}")

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Load pairs of (video_dir_path, label_int).
        """
        video_samples = []
        logger.info(f"[Dataset: {self.__class__.__name__}] Scanning video directories...")
        for label_dir_name, label_int in self.label_map.items():
            label_dir_path = self.split_dir / label_dir_name
            if not label_dir_path.is_dir(): continue

            for video_dir in label_dir_path.iterdir():
                if video_dir.is_dir():
                    video_samples.append((video_dir, label_int))
        
        if self.max_videos and len(video_samples) > self.max_videos:
            logger.info(f"[Dataset: {self.__class__.__name__}] Limited to {self.max_videos} videos ({len(video_samples)} found).")
            video_samples = video_samples[:self.max_videos]
            
            # random.shuffle(video_samples) # Optional: shuffle before limiting

        return video_samples

    def _sample_frame_indices(self, num_available_frames: int) -> List[int]:
        if num_available_frames == 0:
            return []
        
        total_length_needed = (self.sequence_length - 1) * self.sampling_stride + 1

        if num_available_frames >= total_length_needed:
            max_start_idx = num_available_frames - total_length_needed
            start_idx = random.randint(0, max_start_idx)
            indices = [start_idx + i * self.sampling_stride for i in range(self.sequence_length)]
        else:
            # Not enough frames, sample with replacement
            indices = np.random.choice(num_available_frames, self.sequence_length, replace=True)
            indices.sort() # Sort to maintain order

        return indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_dir, label = self.samples[idx]

        frame_paths = sorted([p for p in video_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        num_available = len(frame_paths)

        selected_indices = self._sample_frame_indices(num_available)

        if not selected_indices:
            logger.error(f"[Dataset: {self.__class__.__name__}] No frames available for video {video_dir}.")
        
        frame_tensors = []
        for frame_idx in selected_indices:
            frame_path = frame_paths[frame_idx]

            img = self._load_image(frame_path)
            if self.transform:
                img_tensor = self.transform(img) # Output [C, H, W]
            else:
                img_tensor = T.ToTensor()(img)
            frame_tensors.append(img_tensor)

        sequence_tensor_t_first = torch.stack(frame_tensors, dim=0)
        sequence_tensor_c_first = sequence_tensor_t_first.permute(1, 0, 2, 3)

        return sequence_tensor_c_first, label
