import argparse
from pathlib import Path
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_scene(task):
    """
    Process a single scene directory: sample images and copy to destination.
    Args:
        task (tuple): (scene_dir_str, dest_scene_dir_str, percentage, seed, mode)
    Returns:
        tuple: (scene_dir_str, copied_count, total_count)
    """
    scene_dir_str, dest_scene_dir_str, percentage, seed, mode = task
    random.seed(seed)
    scene_dir = Path(scene_dir_str)
    dest_scene_dir = Path(dest_scene_dir_str)
    dest_scene_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(scene_dir.glob("*.png"))
    total = len(images)
    num_samples = int(total * percentage)
    sampled = []

    if num_samples > 0:
        if mode == 'random':
            sampled = random.sample(images, num_samples)
        elif mode == 'continuous':
            # choose a random start index for a continuous block
            max_start = total - num_samples
            start = random.randint(0, max_start) if max_start > 0 else 0
            sampled = images[start:start + num_samples]

    for img_path in sampled:
        shutil.copy2(img_path, dest_scene_dir / img_path.name)

    return (scene_dir_str, len(sampled), total)


def create_subset(root_dir: Path, percentage: float, seed: int = None, workers: int = None, mode: str = 'random') -> None:
    """
    Create a subset of the dataset at root_dir by sampling percentage of images per scene in parallel.

    Args:
        root_dir (Path): Root dataset directory containing train/val/test subfolders.
        percentage (float): Fraction between 0 and 1 representing sample size.
        seed (int): Random seed for reproducibility.
        workers (int): Number of parallel workers.
        mode (str): Sampling mode: 'random' or 'continuous'.
    """
    # Prepare output directory
    pct_int = int(percentage * 100)
    output_dir = root_dir.parent / f"{root_dir.name}_sub_{pct_int}_{mode}"
    print(f"Creating subset directory: {output_dir}")

    # Gather tasks
    tasks = []
    for split in ["train", "val", "test"]:
        for category in ["REAL", "FAKE"]:
            src_category_dir = root_dir / split / category
            if not src_category_dir.is_dir():
                continue
            for scene_dir in src_category_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                dest_scene_dir = output_dir / split / category / scene_dir.name
                tasks.append((str(scene_dir), str(dest_scene_dir), percentage, seed or 0, mode))

    # Execute in parallel with progress bar
    print(f"🔍 Processing {len(tasks)} scene folders with {workers or 'auto'} workers in '{mode}' mode...")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(process_scene, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Scenes processed"):
            result = future.result()
            results.append(result)

    # Print summary
    for scene_dir, copied, total in results:
        print(f"{scene_dir}: copied {copied} of {total} images.")

    print(f"Subset creation complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a parallelized subset of a dataset by sampling images per scene."
    )
    parser.add_argument(
        "root", type=str,
        help="Root directory of the dataset (contains train/val/test subfolders)"
    )
    parser.add_argument(
        "percentage", type=float,
        help="Fraction between 0 and 1 of images to include in the subset"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["random", "continuous"], default="random",
        help="Sampling mode: 'random' for random frames, 'continuous' for contiguous block"
    )

    args = parser.parse_args()
    root_dir = Path(args.root)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    if not (0.0 < args.percentage <= 1.0):
        raise ValueError("Percentage must be between 0 (exclusive) and 1 (inclusive)")

    create_subset(root_dir, args.percentage, seed=args.seed, workers=args.workers, mode=args.mode)


if __name__ == "__main__":
    main()
