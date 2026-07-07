#!/usr/bin/env python3
"""
Run inference on test images and save predictions in YOLO format.
Output: class_id x_center y_center width height confidence
"""

from ultralytics import YOLO
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
TEST_IMAGES = ROOT / "test" / "images"
OUTPUT_DIR = ROOT / "predictions"

# Model path - update this with your trained model
MODEL_PATH = None  # Will auto-find from runs/


def find_best_model():
    """Find the best trained model."""
    runs_dir = ROOT / "runs"
    if runs_dir.exists():
        candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    # Fallback: look for manually specified model
    for p in ROOT.rglob("best.pt"):
        return p
    raise FileNotFoundError("No best.pt found. Please set MODEL_PATH manually.")


def main():
    model_path = MODEL_PATH or find_best_model()
    print(f"Using model: {model_path}")

    # Load model
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")

    # Get test images
    test_images = sorted(TEST_IMAGES.glob("*.jpg"))
    print(f"Test images: {len(test_images)}")

    # Run inference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=1024,
        conf=0.01,  # low confidence threshold to capture all candidates
        iou=0.7,    # NMS IoU threshold
        device=device,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name="test_preds",
        exist_ok=True,
        verbose=False,
    )

    # Results are saved in YOLO format by ultralytics
    # Format: class_id x_center y_center width height confidence
    print(f"\nInference complete. {len(results)} images processed.")
    print(f"Predictions saved to: {OUTPUT_DIR}/test_preds/labels/")

    # Show summary
    total_boxes = 0
    for result in results:
        if result.boxes is not None:
            total_boxes += len(result.boxes)
    print(f"Total detections: {total_boxes}")


if __name__ == "__main__":
    main()
