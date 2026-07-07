#!/usr/bin/env python3
"""
Generate submission files in the required format.
Reads YOLO predictions and converts them to submission format.
Format: class_id x_center y_center width height confidence

The submit/ directory already has 491 template txt files (one per test image).
We overwrite them with our predictions.

If an image has no detections, we write only the header line.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUBMIT_DIR = ROOT / "submit"
PRED_LABELS_DIR = ROOT / "predictions" / "test_preds" / "labels"


def main():
    # Find prediction label files
    pred_files = sorted(PRED_LABELS_DIR.glob("*.txt")) if PRED_LABELS_DIR.exists() else []
    print(f"Prediction files found: {len(pred_files)}")

    # Find submit template files
    submit_files = sorted(SUBMIT_DIR.glob("*.txt"))
    print(f"Submit template files: {len(submit_files)}")

    updated_count = 0
    empty_count = 0

    for submit_file in submit_files:
        img_id = submit_file.stem  # e.g. "00d1b9d4605c48ac"
        pred_file = PRED_LABELS_DIR / f"{img_id}.txt"

        if pred_file.exists():
            # Read predictions
            with open(pred_file) as f:
                lines = [l.strip() for l in f if l.strip()]

            if lines:
                # Parse and reformat: YOLO format is "class x_center y_center w h conf"
                # We need: "class_id x_center y_center width height confidence"
                formatted_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 6:
                        cls, x, y, w, h, conf = parts[:6]
                        formatted_lines.append(f"{cls} {x} {y} {w} {h} {conf}")
                    elif len(parts) == 5:
                        cls, x, y, w, h = parts
                        formatted_lines.append(f"{cls} {x} {y} {w} {h} 1.0")

                with open(submit_file, "w") as f:
                    for line in formatted_lines:
                        f.write(line + "\n")
                updated_count += 1
            else:
                # Empty predictions - keep file with just the header or empty
                # Check if there's content in the pred file
                with open(submit_file, "w") as f:
                    pass  # empty file for no detections
                empty_count += 1
        else:
            # No prediction for this image - write empty
            with open(submit_file, "w") as f:
                pass
            empty_count += 1

    print(f"Updated with detections: {updated_count}")
    print(f"Empty (no detections): {empty_count}")
    print(f"Total submit files: {updated_count + empty_count}")

    # Verify submission format
    print("\nSample submission entries:")
    for sf in submit_files[:5]:
        content = sf.read_text().strip()
        if content:
            print(f"  {sf.name}: {content[:120]}...")


if __name__ == "__main__":
    main()
