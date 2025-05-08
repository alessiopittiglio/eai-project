import argparse
import json
import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, num_frames: int = 16) -> list:
    """
    Extract `num_frames` evenly spaced frames from a video file.

    Args:
        video_path (Path): Path to the video file.
        num_frames (int): Number of frames to extract.

    Returns:
        List of RGB frames as NumPy arrays.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        return frames

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        logger.warning(f"Empty video file: {video_path}")
        cap.release()
        return frames

    count = min(num_frames, length)
    if count < num_frames:
        logger.warning(
            f"Requested {num_frames} frames, but video has {length}. Extracting {count}."
        )

    indices = np.linspace(0, length - 1, count, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {idx} in {video_path}")
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def save_frames(frames: list, output_dir: Path) -> None:
    """
    Save a list of RGB frames as PNG images in the specified directory.

    Args:
        frames (list): List of RGB frames.
        output_dir (Path): Directory where frames will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        file_path = output_dir / f"frame_{i:04d}.png"
        if not cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            logger.warning(f"Failed saving frame {file_path}")


def _process_single_video(task: dict) -> tuple:
    """
    Process a single video: extract frames and save them.

    Returns:
        Tuple of (status, video_id).
    """
    video_path = task['video_path']
    out_dir = task['output_dir']
    num_frames = task['num_frames']
    skip_existing = task['skip_existing']
    video_id = video_path.stem

    if skip_existing and out_dir.exists() and any(out_dir.iterdir()):
        return 'skipped', video_id

    frames = extract_frames(video_path, num_frames)
    if not frames:
        return 'error', video_id

    save_frames(frames, out_dir)
    return 'processed', video_id


def process_dfdc(video_source_dir: Path, metadata_path: Path,
                 output_dir: Path, num_frames: int = 16,
                 skip_existing: bool = True, workers: int = 4) -> None:
    """
    Extract frames from DFDC dataset videos based on metadata.json.
    """
    logger.info("--- DFDC Dataset Processing ---")

    if not metadata_path.is_file():
        logger.error(f"Metadata not found: {metadata_path}")
        return
    if not video_source_dir.is_dir():
        logger.error(f"Videos dir not found: {video_source_dir}")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)
    df = pd.DataFrame.from_dict(metadata, orient='index').reset_index()
    df.rename(columns={'index': 'filename', 'label': 'label_str'}, inplace=True)
    df['label'] = df['label_str'].map({'REAL': 0, 'FAKE': 1}).dropna().astype(int)

    tasks = []
    for _, row in df.iterrows():
        split = row.get('split', 'undefined')
        label = 'REAL' if row['label'] == 0 else 'FAKE'
        vid = row['filename']
        tasks.append({
            'video_path': video_source_dir / vid,
            'output_dir': output_dir / 'dfdc' / split / label / Path(vid).stem,
            'num_frames': num_frames,
            'skip_existing': skip_existing
        })

    summary = {'processed': 0, 'skipped': 0, 'error': 0}
    # Parallel extraction with progress bar
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for status, _ in tqdm(
            executor.map(_process_single_video, tasks),
            total=len(tasks),
            desc="DFDC Videos"
        ):
            summary[status] += 1

    logger.info("--- DFDC Summary ---")
    for key, count in summary.items():
        logger.info(f"{key.title()}: {count}")


def process_ffpp(orig_root: Path, manip_root: Path, output_dir: Path,
                 num_frames: int = 16, val_size: float = 0.15,
                 test_size: float = 0.15, random_state: int = 42,
                 skip_existing: bool = True, workers: int = 4) -> None:
    """
    Extract frames from FF++ dataset, splitting into train/val/test.
    """
    logger.info("--- FF++ Dataset Processing ---")

    if not orig_root.is_dir() or not manip_root.is_dir():
        logger.error("Invalid FF++ directories.")
        return

    records = []
    for p in orig_root.glob('*.mp4'):
        records.append({'path': p, 'label': 0})
    for p in manip_root.glob('*.mp4'):
        records.append({'path': p, 'label': 1})

    df = pd.DataFrame(records)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df['label'], random_state=random_state
    )
    if val_size + test_size > 0:
        rel_val = val_size / (val_size + test_size)
        val_df, test_df = (
            train_test_split(
                temp_df, test_size=(1 - rel_val), stratify=temp_df['label'], random_state=random_state
            ) if len(temp_df) > 1 else (temp_df, pd.DataFrame(columns=df.columns))
        )
    else:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)

    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    logger.info(
        f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    tasks = []
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            vid = row['path']
            label = 'REAL' if row['label'] == 0 else 'FAKE'
            tasks.append({
                'video_path': vid,
                'output_dir': output_dir / 'ffpp' / split_name / label / vid.stem,
                'num_frames': num_frames,
                'skip_existing': skip_existing
            })

    summary = {'processed': 0, 'skipped': 0, 'error': 0}
    # Parallel extraction with progress bar
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for status, _ in tqdm(
            executor.map(_process_single_video, tasks),
            total=len(tasks),
            desc="FF++ Videos"
        ):
            summary[status] += 1

    logger.info("--- FF++ Summary ---")
    for key, count in summary.items():
        logger.info(f"{key.title()}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess DFDC and FF++ video datasets by extracting frames in parallel."
    )
    parser.add_argument('--dataset', choices=['dfdc', 'ffpp'], required=True,
                        help="Dataset to process: 'dfdc' or 'ffpp'.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Root directory to save frames.")
    parser.add_argument('--num_frames', type=int, default=30,
                        help="Frames to extract per video.")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip videos if frames already exist.")
    parser.add_argument('--workers', type=int, default=4,
                        help="Parallel worker processes.")

    # DFDC args
    dfdc_group = parser.add_argument_group('DFDC')
    dfdc_group.add_argument('--dfdc_metadata_path', type=str,
                             default='./data/metadata.json', help="DFDC metadata.json")
    dfdc_group.add_argument('--dfdc_video_dir', type=str,
                             default='./data/dfdc_videos', help="DFDC video files dir.")

    # FF++ args
    ffpp_group = parser.add_argument_group('FF++')
    ffpp_group.add_argument('--ffpp_orig_root', type=str,
                             default='./data/ff++/original_sequences/actors/c23/videos', help="Original videos dir.")
    ffpp_group.add_argument('--ffpp_manip_root', type=str,
                             default='./data/ff++/manipulated_sequences/DeepFakeDetection/c23/videos', help="Manipulated videos dir.")
    ffpp_group.add_argument('--ffpp_val_size', type=float, default=0.15,
                             help="Validation ratio.")
    ffpp_group.add_argument('--ffpp_test_size', type=float, default=0.15,
                             help="Test ratio.")
    ffpp_group.add_argument('--ffpp_random_state', type=int, default=42,
                             help="Random seed.")

    args = parser.parse_args()
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'dfdc':
        process_dfdc(
            video_source_dir=Path(args.dfdc_video_dir),
            metadata_path=Path(args.dfdc_metadata_path),
            output_dir=out_path,
            num_frames=args.num_frames,
            skip_existing=args.skip_existing,
            workers=args.workers
        )
    else:
        process_ffpp(
            orig_root=Path(args.ffpp_orig_root),
            manip_root=Path(args.ffpp_manip_root),
            output_dir=out_path,
            num_frames=args.num_frames,
            val_size=args.ffpp_val_size,
            test_size=args.ffpp_test_size,
            random_state=args.ffpp_random_state,
            skip_existing=args.skip_existing,
            workers=args.workers
        )

    logger.info("Preprocessing completed.")
