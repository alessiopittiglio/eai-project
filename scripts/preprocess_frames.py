import argparse
import json
import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We moved here the function to extract frames from a video file.

def extract_frames(video_path, num_frames=16):
    """
    Extract `num_frames` evenly spaced frames from a video.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if length < 1:
        print(f"Warning: Video file {video_path} seems to be empty.")
        cap.release()
        return frames
    
    actual_num_frames = min(num_frames, length)

    if actual_num_frames < num_frames:
        print(f"Warning: Asking for {num_frames} frames, but video has only {length} frames. Extracting {actual_num_frames} frames instead.")

    frame_idxs = list(np.linspace(0, length - 1, actual_num_frames).astype(int))

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read frame {idx} from video {video_path}.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames

def save_frames(frames, output_dir):
    """
    Saves a list of NumPy frames (RGB) to an output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_rgb in enumerate(frames):
        frame_filename = output_dir / f"frame_{i:04d}.png"
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(frame_filename), frame_bgr)
        if not success:
            logger.warning(f"Failed to save frame: {frame_filename}.")
        
def process_dfdc(video_source_dir, metadata_path, output_dir, num_frames, skip_existing):
    """
    Processes DFDC dataset videos based on metadata.json.
    """
    logger.info("--- Processing DFDC Dataset ---")
    if not metadata_path.is_file():
        logger.error(f"DFDC metadata file not found: {metadata_path}")
        return
    if not video_source_dir.is_dir():
        logger.error(f"DFDC video source directory not found: {video_source_dir}")
        return
    
    logger.info(f"Loading metadata from: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    metadata_df = pd.DataFrame(metadata).transpose().reset_index()
    metadata_df.rename(columns={'index': 'filename', 'label': 'label_str'}, inplace=True)
    metadata_df['label'] = metadata_df['label_str'].map({'REAL': 0, 'FAKE': 1})
    metadata_df = metadata_df.dropna(subset=['label'])
    metadata_df['label'] = metadata_df['label'].astype(int)

    logger.info(f"Found {len(metadata_df)} entries in metadata.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing DFDC Videos"):
        video_filename = row['filename']
        label = row['label']
        split = row['split'] # 'train', 'val', 'test'
        label_name = "REAL" if label == 0 else "FAKE"
        video_id = Path(video_filename).stem

        target_frame_dir = output_dir / "dfdc" / split / label_name / video_id
        if skip_existing and target_frame_dir.exists() and any(target_frame_dir.iterdir()):
            skipped_count += 1
            continue

        video_path = video_source_dir / video_filename
        if not video_path.is_file():
            #logger.warning(f"Video file not found: {video_path}")
            error_count += 1
            continue

        frames = extract_frames(video_path, num_frames)
        if not frames:
            #logger.warning(f"No frames extracted from video: {video_path}")
            error_count += 1
            continue

        save_frames(frames, target_frame_dir)
        processed_count += 1

        logger.info("--- DFDC Processing Summary ---")
        logger.info(f"Successfully processed: {processed_count} videos")
        logger.info(f"Skipped (already exist): {skipped_count} videos")
        logger.info(f"Errors (file not found/extraction failed): {error_count} videos")

def process_ffpp(orig_root, manip_root, output_dir, num_frames, val_size, test_size, random_state, skip_existing):
    """
    Processes FF++ dataset videos, performing split and frame extraction.
    """
    logger.info("--- Processing FF++ Dataset ---")
    if not orig_root.is_dir():
        logger.error(f"FF++ original videos directory not found: {orig_root}")
        return
    if not manip_root.is_dir():
        logger.error(f"FF++ manipulated videos directory not found: {manip_root}")
        return
    
    # We moved here the logic to gather video paths
    logger.info("Gathering FF++ video paths...")
    orig_paths = list(orig_root.glob("*.mp4"))
    manip_paths = list(manip_root.glob("*.mp4"))
    if not orig_paths and not manip_paths:
        logger.error(f"No .mp4 videos found in {orig_root} or {manip_root}")
        return
    
    records = []
    for p in orig_paths:
        records.append({"filename": p.name, "label": 0, "path": p}) # label 0 = REAL/original
    for p in manip_paths:
        records.append({"filename": p.name, "label": 1, "path": p}) # label 1 = FAKE/manipulated
    df = pd.DataFrame.from_records(records)
    logger.info(f"Found {len(orig_paths)} original and {len(manip_paths)} manipulated videos.")

    logger.info(f"Splitting data: val_size={val_size}, test_size={test_size}")
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size),
        stratify=df["label"], random_state=random_state
    )
    if val_size + test_size > 0:
        rel_val = val_size / (val_size + test_size)
        if len(temp_df) > 1 and rel_val > 0 and rel_val < 1:
            val_df, test_df = train_test_split(
                temp_df, test_size=max(0.0, (1.0 - rel_val)),
                stratify=temp_df["label"], random_state=random_state
            )
        elif len(temp_df) > 0 and rel_val == 1.0:
            val_df = temp_df
            test_df = pd.DataFrame(columns=temp_df.columns)
        elif len(temp_df) > 0 and rel_val == 0.0:
            test_df = temp_df
            val_df = pd.DataFrame(columns=temp_df.columns)
        else:
            val_df = temp_df
            test_df = pd.DataFrame(columns=temp_df.columns)
    else:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for split_name, split_df in splits.items():
        logger.info(f"Processing split: {split_name} ({len(split_df)} videos)")
        if split_df.empty:
            continue
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            video_path = row["path"]
            label = row["label"]
            video_id = video_path.stem
            label_name = "REAL" if label == 0 else "FAKE"

            target_frame_dir = output_dir / "ffpp" / split_name / label_name / video_id
            if skip_existing and target_frame_dir.exists() and any(target_frame_dir.iterdir()):
                # logger.info(f"Skipping {video_id} - frames already exist.")
                skipped_count += 1
                continue

            frames = extract_frames(video_path, num_frames)
            if not frames:
                # logger.warning(f"No frames extracted for {video_path}. Skipping.")
                error_count += 1
                continue

            save_frames(frames, target_frame_dir)
            processed_count += 1

    logger.info("--- FF++ Processing Summary ---")
    logger.info(f"Successfully processed: {processed_count} videos")
    logger.info(f"Skipped (already exist): {skipped_count} videos")
    logger.info(f"Errors (extraction failed): {error_count} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video datasets (DFDC, FF++) by extracting frames.")

    parser.add_argument('--dataset', type=str, required=True, choices=['dfdc', 'ffpp'],
                        help="Name of the dataset to process.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Root directory to save the extracted frames.")
    parser.add_argument('--num_frames', type=int, default=30,
                        help="Number of frames to extract evenly from each video.")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip processing videos if their frame directory already exists and is not empty.")

    # DFDC specific arguments
    dfdc_group = parser.add_argument_group("DFDC Arguments")
    dfdc_group.add_argument('--dfdc_metadata_path', type=str, default='./data/metadata.json',
                            help="Path to the DFDC metadata.json file.")
    dfdc_group.add_argument('--dfdc_video_dir', type=str, default='./data/dfdc_videos',
                           help="Directory containing ALL DFDC video files (.mp4).") # I don't know if this is the right path

    # FF++ specific arguments
    ffpp_group = parser.add_argument_group("FF++ Arguments")
    ffpp_group.add_argument('--ffpp_orig_root', type=str, default='./data/ffpp/original_sequences/youtube/c23/videos',
                            help="Directory containing original FF++ videos.")
    ffpp_group.add_argument('--ffpp_manip_root', type=str, default='./data/ffpp/manipulated_sequences/Deepfakes/c23/videos',
                           help="Directory containing manipulated FF++ videos (e.g., Deepfakes).")
    ffpp_group.add_argument('--ffpp_val_size', type=float, default=0.15,
                           help="Proportion of data for the validation set (for FF++ split).")
    ffpp_group.add_argument('--ffpp_test_size', type=float, default=0.15,
                           help="Proportion of data for the test set (for FF++ split).")
    ffpp_group.add_argument('--ffpp_random_state', type=int, default=42,
                           help="Random state for FF++ train/val/test splitting.")
    
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == "dfdc":
        process_dfdc(
            video_source_dir=Path(args.dfdc_video_dir),
            metadata_path=Path(args.dfdc_metadata_path),
            output_dir=output_path,
            num_frames=args.num_frames,
            skip_existing=args.skip_existing
        )
    elif args.dataset == "ffpp":
        process_ffpp(
            orig_root=Path(args.ffpp_orig_root),
            manip_root=Path(args.ffpp_manip_root),
            output_dir=output_path,
            num_frames=args.num_frames,
            val_size=args.ffpp_val_size,
            test_size=args.ffpp_test_size,
            random_state=args.ffpp_random_state,
            skip_existing=args.skip_existing
        )

    logger.info("Preprocessing completed.")
