import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

def check_single_image(path_str: str) -> tuple[str, bool, str]:
    path = Path(path_str)
    try:
        with Image.open(path) as img:
            img.convert("RGB")
        return (str(path), True, "")
    except (OSError, UnidentifiedImageError, ValueError) as e:
        return (str(path), False, str(e))

def check_images_parallel(root_dir: Path, delete: bool = False, workers: int = None):
    image_paths = list(map(str, root_dir.rglob("*.png")))
    print(f"🔍 Scanning {len(image_paths)} images...")

    corrupted = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_single_image, path): path for path in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking images"):
            path, ok, err = future.result()
            if not ok:
                corrupted.append((path, err))

    if not corrupted:
        print("\n✅ No corrupted images found.")
        return

    print(f"\n🚫 Found {len(corrupted)} corrupted images:")
    for path, err in corrupted:
        print(f"{path} — {err}")

    if delete:
        print("\n🧹 Deleting corrupted images...")
        for path, _ in tqdm(corrupted, desc="Deleting"):
            try:
                Path(path).unlink()
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
        print("✅ Deletion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel check and optional deletion of corrupted .png images.")
    parser.add_argument("root", type=str, help="Root directory containing train/val/test subfolders")
    parser.add_argument("--delete", action="store_true", help="Delete corrupted images")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    check_images_parallel(Path(args.root), delete=args.delete, workers=args.workers)
