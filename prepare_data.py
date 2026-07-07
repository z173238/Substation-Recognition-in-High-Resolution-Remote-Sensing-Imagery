#!/usr/bin/env python3
"""
Split training data into train/val sets for YOLO training.
Creates train_split/ and val_split/ directories with symlinks to images and labels.
"""

import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_IMAGES = ROOT / "train" / "images"
TRAIN_LABELS = ROOT / "train" / "labels"
TRAIN_SPLIT = ROOT / "train_split"
VAL_SPLIT = ROOT / "val_split"
VAL_RATIO = 0.15  # 15% for validation
SEED = 42


def main():
    random.seed(SEED)

    # Get all images that have labels
    image_files = sorted(TRAIN_IMAGES.glob("*.jpg"))
    print(f"Total training images with labels: {len(image_files)}")

    # Shuffle and split
    random.shuffle(image_files)
    val_count = max(1, int(len(image_files) * VAL_RATIO))
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create directories
    for split_dir, files, split_name in [
        (TRAIN_SPLIT, train_files, "train"),
        (VAL_SPLIT, val_files, "val"),
    ]:
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Remove old files
        for old in img_dir.iterdir():
            old.unlink()
        for old in lbl_dir.iterdir():
            old.unlink()

        # Symlink images
        for img_path in files:
            dst = img_dir / img_path.name
            if not dst.exists():
                os.symlink(img_path.resolve(), dst)

        # Symlink labels
        for img_path in files:
            lbl_name = img_path.stem + ".txt"
            src_lbl = TRAIN_LABELS / lbl_name
            dst_lbl = lbl_dir / lbl_name
            if src_lbl.exists() and not dst_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)

        print(f"Created {split_name} split: {len(list(img_dir.iterdir()))} images, "
              f"{len(list(lbl_dir.iterdir()))} labels")


if __name__ == "__main__":
    main()
