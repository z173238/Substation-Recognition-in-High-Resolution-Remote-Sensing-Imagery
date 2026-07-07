#!/usr/bin/env python3
"""
Optimize inference for competition submission.
1. Tune confidence threshold on validation set
2. Run TTA inference on test set
3. Optionally ensemble multiple checkpoints
"""

import csv
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "substation_yolo11m"
WEIGHTS_DIR = RUNS_DIR / "weights"
DATA_YAML = ROOT / "data.yaml"


def tune_confidence_threshold(model_path, data_yaml, device=0):
    """Find the best confidence threshold on validation set."""
    print("=" * 60)
    print("Step 1: Tuning confidence threshold on validation set")
    print("=" * 60)

    model = YOLO(str(model_path))

    # Run validation with different conf thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    results_table = []

    for conf in thresholds:
        metrics = model.val(
            data=str(data_yaml),
            conf=conf,
            iou=0.7,
            device=device,
            verbose=False,
            plots=False,
        )
        mAP50 = metrics.box.map50
        mAP50_95 = metrics.box.map
        results_table.append((conf, mAP50, mAP50_95))
        print(f"  conf={conf:.3f}: mAP50={mAP50:.4f}  mAP50-95={mAP50_95:.4f}")

    # Find best
    best_50 = max(results_table, key=lambda x: x[1])
    best_95 = max(results_table, key=lambda x: x[2])
    # fitness = 0.1*mAP50 + 0.9*mAP50-95
    best_fitness = max(results_table, key=lambda x: 0.1 * x[1] + 0.9 * x[2])

    print(f"\n  Best mAP50:    conf={best_50[0]:.3f} -> {best_50[1]:.4f}")
    print(f"  Best mAP50-95: conf={best_95[0]:.3f} -> {best_95[2]:.4f}")
    print(f"  Best fitness:  conf={best_fitness[0]:.3f} -> mAP50={best_fitness[1]:.4f} mAP50-95={best_fitness[2]:.4f}")

    return best_fitness[0], results_table


def run_tta_inference(model_path, test_dir, output_dir, conf_threshold, device=0):
    """Run TTA (Test Time Augmentation) inference on test set."""
    print("\n" + "=" * 60)
    print("Step 2: TTA Inference on test set")
    print("=" * 60)

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(test_dir),
        imgsz=1024,
        conf=conf_threshold,
        iou=0.7,
        device=device,
        augment=True,   # TTA: horizontal flip + scale variations
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="tta_preds",
        exist_ok=True,
        verbose=False,
    )

    total_boxes = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"  Test images: {len(results)}")
    print(f"  Total detections: {total_boxes}")
    print(f"  Avg per image: {total_boxes / len(results):.1f}")
    return results


def run_standard_inference(model_path, test_dir, output_dir, conf_threshold, device=0):
    """Run standard (non-TTA) inference."""
    model = YOLO(str(model_path))

    results = model.predict(
        source=str(test_dir),
        imgsz=1024,
        conf=conf_threshold,
        iou=0.7,
        device=device,
        augment=False,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="std_preds",
        exist_ok=True,
        verbose=False,
    )

    total_boxes = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"  Test images: {len(results)}")
    print(f"  Total detections: {total_boxes}")
    print(f"  Avg per image: {total_boxes / len(results):.1f}")
    return results


def generate_submit(pred_labels_dir, submit_dir, tag=""):
    """Generate submission files from predictions."""
    pred_dir = Path(pred_labels_dir)
    submit_d = Path(submit_dir) if tag == "" else Path(str(submit_dir) + tag)

    pred_files = {f.stem: f for f in pred_dir.glob("*.txt")}
    submit_d.mkdir(parents=True, exist_ok=True)

    # Clean old files in submit dir
    for old in submit_d.glob("*.txt"):
        old.unlink()

    total_boxes = 0
    for submit_file in sorted(Path(ROOT / "submit").glob("*.txt")):
        img_id = submit_file.stem
        pred_file = pred_files.get(img_id)

        out_file = submit_d / f"{img_id}.txt"
        if pred_file and pred_file.exists():
            with open(pred_file) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                # Keep only class_id x_center y_center width height confidence
                formatted = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 6:
                        formatted.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}")
                    elif len(parts) == 5:
                        formatted.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} 1.0")

                with open(out_file, "w") as f:
                    for line in formatted:
                        f.write(line + "\n")
                total_boxes += len(formatted)
            else:
                out_file.write_text("")
        else:
            out_file.write_text("")

    print(f"  Files: {len(list(submit_d.glob('*.txt')))}")
    print(f"  Total boxes: {total_boxes}")
    return submit_d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Tune confidence threshold")
    parser.add_argument("--tta", action="store_true", help="Run TTA inference")
    parser.add_argument("--std", action="store_true", help="Run standard inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()
    if len([x for x in [args.tune, args.tta, args.std, args.all] if x]) == 0:
        args.all = True  # default: run everything

    device = 0 if torch.cuda.is_available() else "cpu"
    model_path = args.model or str(WEIGHTS_DIR / "best.pt")
    test_dir = ROOT / "test" / "images"
    output_base = ROOT / "predictions_opt"
    conf = args.conf

    if args.tune or args.all:
        conf, _ = tune_confidence_threshold(model_path, DATA_YAML, device)

    if args.tta or args.all:
        tta_output = output_base / "tta"
        run_tta_inference(model_path, test_dir, tta_output, conf, device)
        generate_submit(tta_output / "tta_preds" / "labels", ROOT / "submit_tta", "_tta")

    if args.std or args.all:
        std_output = output_base / "std"
        run_standard_inference(model_path, test_dir, std_output, conf, device)
        generate_submit(std_output / "std_preds" / "labels", ROOT / "submit_opt", "")

    print("\n" + "=" * 60)
    print("Done! Optimized submissions:")
    if args.tta or args.all:
        print(f"  TTA version: submit_tta/")
    if args.std or args.all:
        print(f"  Standard version: submit_opt/")
    print(f"  Confidence threshold used: {conf:.3f}")
