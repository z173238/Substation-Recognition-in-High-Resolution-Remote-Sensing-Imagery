#!/usr/bin/env python3
"""Optimized inference: low conf threshold + TTA for better mAP."""
from ultralytics import YOLO
from pathlib import Path
import time

ROOT = Path('/home/ubuntu/workspace/20260706_SiChuan_MeiShan_Substation_Competition')
MODEL = str(ROOT / 'runs/substation_yolo11m/weights/best.pt')
TEST = str(ROOT / 'test/images')
CONF = 0.001  # ultra-low to maximize recall

def run(name, augment, conf):
    model = YOLO(MODEL)
    t0 = time.time()
    results = model.predict(
        source=TEST, imgsz=1024, conf=conf, iou=0.7,
        device='cpu', augment=augment, save=False,
        save_txt=True, save_conf=True,
        project=str(ROOT / 'predictions_opt'),
        name=name, exist_ok=True, verbose=False,
    )
    boxes = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    elapsed = time.time() - t0
    print(f'{name}: {len(results)} imgs, {boxes} boxes, {elapsed:.0f}s ({elapsed/len(results):.1f}s/img)')
    return str(ROOT / 'predictions_opt' / name / 'labels')

def make_submit(labels_dir, output_dir):
    labels = Path(labels_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.txt'): old.unlink()

    total = 0
    for sf in sorted((ROOT / 'submit').glob('*.txt')):
        pred = labels / f'{sf.stem}.txt'
        if pred.exists():
            lines = [l.strip() for l in pred.read_text().strip().split('\n') if l.strip()]
            formatted = []
            for line in lines:
                p = line.split()
                if len(p) >= 6:
                    formatted.append(f'{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}')
            (out / sf.name).write_text('\n'.join(formatted) + '\n' if formatted else '')
            total += len(formatted)
        else:
            (out / sf.name).write_text('')
    print(f'  -> {len(list(out.glob("*.txt")))} files, {total} boxes -> {output_dir}')

# ---- Standard (no TTA) ----
print('=== Standard inference (low conf) ===')
labels_std = run('std_lowconf', augment=False, conf=CONF)

# ---- TTA ----
print('\n=== TTA inference (low conf, augment=True) ===')
labels_tta = run('tta_lowconf', augment=True, conf=CONF)

# ---- Generate submissions ----
print('\n=== Generating submissions ===')
make_submit(labels_std, str(ROOT / 'submit_opt'))
make_submit(labels_tta, str(ROOT / 'submit_tta'))

print('\nDone! Optimized results:')
print(f'  submit_opt/  - standard, conf={CONF}')
print(f'  submit_tta/  - TTA, conf={CONF}')
