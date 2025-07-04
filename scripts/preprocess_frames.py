import argparse
from collections import defaultdict
import json
import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, num_frames: int = 16) -> list:
    """
    Extract `num_frames` evenly spaced frames from a video file.
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        file_path = output_dir / f"frame_{i:04d}.png"
        if not cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            logger.warning(f"Failed saving frame {file_path}")


def _process_single_video(task: dict) -> tuple:
    """
    Process a single video: extract frames and save them.
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


def random_augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply a random augmentation to a single RGB image.
    """
    # brightness jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = np.random.uniform(0.8, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # horizontal flip
    if np.random.rand() < 0.5:
        img_aug = np.fliplr(img_aug)

    # rotation
    angle = np.random.uniform(-15, 15)
    h, w = img_aug.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_aug = cv2.warpAffine(img_aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return img_aug


def augment_video_dir(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy + augment all frames from `src_dir` → `dst_dir` once.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fpath in sorted(src_dir.iterdir()):
        if fpath.suffix.lower() not in {'.png','.jpg','.jpeg'}:
            continue
        img = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        img_aug = random_augment_image(img)
        out_bgr = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
        out_path = dst_dir / fpath.name
        if not cv2.imwrite(str(out_path), out_bgr):
            logger.warning(f"Failed saving augmented frame {out_path}")


def _augment_task(task: dict) -> tuple:
    """
    Single augmentation job: skip or perform.
    Returns ('skipped'|'augmented', dst_dir).
    """
    src = task['src_dir']
    dst = task['dst_dir']
    skip_existing = task['skip_existing']

    if skip_existing and dst.exists() and any(dst.iterdir()):
        return 'skipped', dst

    augment_video_dir(src, dst)
    return 'augmented', dst


def augment_for_balance(dataset_root: Path, skip_existing: bool, workers: int) -> None:
    """
    Build and run augmentation tasks in parallel so that each label
    under each split of `dataset_root` is brought up to the max count.
    """
    logger.info(f"--- Augmentation for balance at {dataset_root} ---")
    tasks = []

    # gather jobs
    for split_dir in sorted(dataset_root.iterdir()):
        if not split_dir.is_dir():
            continue

        # count how many sample‐dirs per label
        label_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        counts = {d.name: len([c for c in d.iterdir() if c.is_dir()]) for d in label_dirs}
        if not counts:
            continue

        max_count = max(counts.values())
        for label_dir in label_dirs:
            label = label_dir.name
            count = counts[label]
            if count >= max_count:
                logger.info(f"{split_dir.name}/{label}: {count} samples (no augmentation needed)")
                continue

            needed = max_count - count
            logger.info(f"{split_dir.name}/{label}: {count} samples; will augment {needed} more to reach {max_count}")

            existing = [d for d in label_dir.iterdir() if d.is_dir()]
            for i in range(needed):
                src = existing[np.random.randint(len(existing))]
                aug_name = f"{src.stem}_aug{i+1:03d}"
                dst = label_dir / aug_name
                logger.debug(f"  Queuing augmentation {split_dir.name}/{label}: {src.stem} → {aug_name}")
                tasks.append({
                    'src_dir': src,
                    'dst_dir': dst,
                    'skip_existing': skip_existing
                })

    # run in parallel
    summary = defaultdict(int)
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for status, dst in tqdm(exe.map(_augment_task, tasks), total=len(tasks), desc="Augmenting"):
            summary[status] += 1
            logger.info(f"Augmentation {status}: {dst}")

    # report
    logger.info("--- Augmentation Summary ---")
    for key, cnt in summary.items():
        logger.info(f"{key.title()}: {cnt}")
    logger.info(f"--- Done augmenting {dataset_root} ---")

def process_celebdfv2(root_dir: Path, output_dir: Path, 
                      num_frames: int = 16, 
                      skip_existing: bool = True, 
                      workers: int = 4) -> None:
    """
    Process Celeb-DF-v2 dataset extracting frames from all videos
    in Celeb-real, Celeb-synthesis, YouTube-real folders.
    """
    logger.info("--- Celeb-DF-v2 Dataset Processing ---")
    
    test_list = (root_dir / "List_of_testing_videos.txt").read_text().splitlines()
    test_set = set(line.split(' ', 1)[1] for line in test_list)

    folders = {
        'Celeb-real': 0,
        'Celeb-synthesis': 1,
        'YouTube-real': 0,
    }

    tasks = []
    for folder_name, label in folders.items():
        folder_path = root_dir / folder_name
        if not folder_path.is_dir():
            logger.warning(f"Missing folder: {folder_path}")
            continue
        for video_path in folder_path.glob("*.mp4"):
            rel_path = f"{folder_name}/{video_path.name}"
            if rel_path not in test_set:
                continue  # usa solo test set, o rimuovi questo controllo per tutto il dataset

            split = "test"
            label_name = "REAL" if label == 0 else "FAKE"
            out_dir = output_dir / 'celebdfv2' / split / label_name / video_path.stem

            tasks.append({
                'video_path': video_path,
                'output_dir': out_dir,
                'num_frames': num_frames,
                'skip_existing': skip_existing,
            })
        
    summary = {'processed': 0, 'skipped': 0, 'error': 0}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for status, _ in tqdm(executor.map(_process_single_video, tasks), total=len(tasks), desc="CelebDF-V2 Videos"):
            summary[status] += 1

    logger.info("--- Celeb-DF-v2 Summary ---")
    for key, count in summary.items():
        logger.info(f"{key.title()}: {count}")


def process_dfdc(video_source_dir: Path, 
                 metadata_path: Path,
                 output_dir: Path, 
                 num_frames: int = 16,
                 skip_existing: bool = True, 
                 workers: int = 4,
                 augment: bool = False,
                 ) -> None:
    """
    Extract frames from DFDC dataset videos based on metadata.json,
    then augment to balance classes.
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

    if augment:
        # now balance by augmenting
        augment_for_balance(output_dir / 'dfdc', skip_existing=skip_existing, workers=workers)


def process_ffpp(orig_root: Path, 
                 manip_root: Path, 
                 output_dir: Path,
                 num_frames: int = 16, 
                 val_size: float = 0.15,
                 test_size: float = 0.15,
                 random_state: int = 42,
                 skip_existing: bool = True, 
                 workers: int = 4,
                augment: bool = False,
                 ) -> None:
    """
    Extract frames from FF++ dataset, split into train/val/test,
    then augment to balance classes in each split.
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
    logger.info(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

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

    if augment:
        # now balance by augmenting
        augment_for_balance(output_dir / 'ffpp', skip_existing=skip_existing, workers=workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess DFDC and FF++ video datasets by extracting frames in parallel."
    )
    parser.add_argument('--dataset', choices=['dfdc', 'ffpp', 'celebdfv2'], required=True,
                        help="Dataset to process: 'dfdc' or 'ffpp'.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Root directory to save frames.")
    parser.add_argument('--num_frames', type=int, default=30,
                        help="Frames to extract per video.")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip videos if frames already exist.")
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help="Parallel worker processes (default: number of CPU cores).")
    parser.add_argument("--augment", action="store_true",)

    # DFDC args
    dfdc_group = parser.add_argument_group('DFDC')
    dfdc_group.add_argument('--dfdc_metadata_path', type=str,
                             default='./data/DeepFakeDataset/metadata.json', help="DFDC metadata.json")
    dfdc_group.add_argument('--dfdc_video_dir', type=str,
                             default='./data/DeepFakeDataset', help="DFDC video files dir.")

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
    
    # Celeb-DF-v2 args
    celebdfv2_group = parser.add_argument_group('Celeb-DF-v2')
    celebdfv2_group.add_argument('--celebdfv2_root', type=str,
                                 default='./data/Celeb-DF-v2', help="Celeb-DF-v2 root directory.")

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
            workers=args.workers,
            augment=args.augment
        )
    elif args.dataset == 'celebdfv2':
        process_celebdfv2(
            root_dir=Path(args.celebdfv2_root),
            output_dir=out_path,
            num_frames=args.num_frames,
            skip_existing=args.skip_existing,
            workers=args.workers,
        )
    elif args.dataset == 'ffpp':
        process_ffpp(
            orig_root=Path(args.ffpp_orig_root),
            manip_root=Path(args.ffpp_manip_root),
            output_dir=out_path,
            num_frames=args.num_frames,
            val_size=args.ffpp_val_size,
            test_size=args.ffpp_test_size,
            random_state=args.ffpp_random_state,
            skip_existing=args.skip_existing,
            workers=args.workers,
            augment=args.augment
        )

    logger.info("Preprocessing completed.")
