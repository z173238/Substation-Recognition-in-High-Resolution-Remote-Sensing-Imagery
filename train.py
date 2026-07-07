#!/usr/bin/env python3
"""
Train YOLO model for substation detection.
Uses the yolo_env conda environment.
Run: conda activate yolo_env && python train.py
"""

from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"

# Training configuration
MODEL_PATH = "/home/ubuntu/workspace/ultralytics-zsl-wood/yolo11m.pt"  # Use locally cached model
MODEL_NAME = "yolo11m"
EPOCHS = 200
IMG_SIZE = 1024
BATCH = 16
DEVICE = 0  # GPU device


def main():
    print(f"Data config: {DATA_YAML}")
    print(f"Using model: {MODEL_PATH}")

    # Load a pretrained model from local file
    model = YOLO(MODEL_PATH)

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=8,
        amp=False,  # disable AMP to avoid network download during check
        cache=True,  # cache images in RAM for faster training
        patience=30,  # early stopping
        save=True,
        save_period=10,
        project=str(ROOT / "runs"),
        name=f"substation_{MODEL_NAME}",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=90.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print(f"\nTraining complete. Best model: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
