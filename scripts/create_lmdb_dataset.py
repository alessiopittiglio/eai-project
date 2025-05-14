import argparse
import lmdb
import logging
import random
import tqdm
import pickle
from pathlib import Path

format = "'%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger(__name__)

def create_lmdb_for_split(
        frames_root_dir: Path, # i.e. /data_frames/ffpp/train
        output_lmdb_path: Path,
        label_dirs_map: dict, # i.e. {'real': 0, 'fake': 1}
        write_frequency: int = 5000,
        max_size_gb: int = 100
    ):
    """
    Creates an LMDB database for a specific split (train, val, test) containing images
    and their labels.
    Keys will be unique strings (e.g., "real/video_id/frame_0000").
    Values will be PNG/JPG image bytes. Labels will be saved separately or included.
    """
    if output_lmdb_path.exists():
        logger.warning(
            f"The LMDB database {output_lmdb_path} already exists."
            "It will be overwritten."
        )
        for item in output_lmdb_path.iterdir():
            item.unlink()
        output_lmdb_path.rmdir()

    output_lmdb_path.mkdir(parents=True, exist_ok=True)
    map_size = max_size_gb * 1024 * 1024 * 1024 # GB to bytes

    logger.info(
        f"Creating LMDB database at: {output_lmdb_path} with map_size: {max_size_gb} GB"
    )
    env = lmdb.open(str(output_lmdb_path), map_size=map_size)

    all_frame_infos = []
    for label_name, label_int in label_dirs_map.items():
        label_dir = frames_root_dir / label_name
        if not label_dir.is_dir():
            logger.warning(
                f"Label directory {label_dir} not found for the split. Skipping."
            )
            continue
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir(): continue
            for frame_file in video_dir.glob('*'):
                if frame_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Create a unique key for this frame
                    # Example: real_video001_frame0000 (if video_id is a string)
                    
                    key_str = f"{label_name}/{video_dir.name}/{frame_file.stem}"
                    all_frame_infos.append(
                        {
                            'key': key_str.encode('utf-8'), 
                            'path': frame_file,
                            'label': label_int
                        }
                    )
    if not all_frame_infos:
        logger.warning(f"No frames found in {frames_root_dir}. LMDB will be empty.")
        env.close()
        return
    
    # Shuffle all frame information before writing to LMDB.
    # This can potentially lead to a more balanced B-tree structure within LMDB
    # and distribute writes more evenly across database pages, though the
    # primary shuffling for training is handled by the DataLoader.
    random.shuffle(all_frame_infos)
    logger.info(f"Found {len(all_frame_infos)} frames to write into the LMDB.")

    with env.begin(write=True) as txn:
        keys_list = []
        for i, frame_info in enumerate(tqdm(all_frame_infos, desc="Writing to LMDB")):
            try:
                with open(frame_info['path'], 'rb') as f:
                    image_bytes = f.read()

                txn.put(frame_info['key'], image_bytes)
                keys_list.append(frame_info['key'])

                label_key = (
                    frame_info['key'].decode('utf-8') + "_label"
                ).encode('utf-8')
                txn.put(label_key, pickle.dumps(frame_info['label']))

                if (i + 1) % write_frequency == 0:
                    logger.info(
                        f"Written {i + 1}/{len(all_frame_infos)} items into the DB"
                    )
            except Exception as e:
                logger.error(f"Error writing {frame_info['path']} to the DB: {e}")

        logger.info("Saving list of keys and labels into the DB...")
        all_keys_with_labels = [
            {
                'key': fi['key'], 
                'label': fi['label']
            } for fi in all_frame_infos
        ]
        txn.put(b'__keys_with_labels__', pickle.dumps(all_keys_with_labels))
    
    logger.info(
        f"LMDB database successfully created for {frames_root_dir}"
        f"at {output_lmdb_path}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create LMDB dataset from pre-extracted frames."
    )

    parser.add_argument(
        '--input_frames_root',
        type=str,
        required=True,
        help="Root directory of the pre-extracted frames (e.g., ./data_frames/ffpp)."
    )
    parser.add_argument(
        '--output_lmdb_root', 
        type=str, 
        required=True,
        help="Root directory where LMDB databases will be created (e.g., ./data_lmdb)."
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True, 
        choices=['ffpp', 'dfdc'],
        help="Name of the dataset (used for subdirectories)."
    )
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help="List of splits to process (e.g., train val test).")
    parser.add_argument('--max_size_gb_per_split', type=int, default=50,
                        help="Estimated max size in GB for each LMDB split database.")
    
    args = parser.parse_args()

    output_lmdb_dataset_dir = Path(args.output_lmdb_root) / args.dataset_name

    label_map = {'real': 0, 'fake': 1}

    for split_name in args.splits:
        split_frames_dir = Path(args.input_frames_root) / args.dataset_name / split_name
        split_lmdb_dir = output_lmdb_dataset_dir / split_name

        if not split_frames_dir.is_dir():
            logger.warning(
                f"Frame directory for split '{split_name}' not found: " 
                f"{split_frames_dir}. Skipping."
            )
            continue
        
        logger.info(f"Creating LMDB for {args.dataset_name} - Split: {split_name}")
        create_lmdb_for_split(
            frames_root_dir=split_frames_dir,
            output_lmdb_path=split_lmdb_dir,
            label_dirs_map=label_map,
            max_size_gb=args.max_size_gb_per_split
        )
    logger.info("All LMDB splits created.")
 