"""
Find Best Minority Class Images for Annotation
================================================
Scans data/v7_upload/ (2,953 new unannotated images)
and selects the best ones containing minority classes.

Run from: urban-digital-twin-backend/
Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/find_minority_images.py
"""

import os, shutil
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

MODEL_PATH = "runs/detect/newcastle_v6_improved/weights/best.pt"
INPUT_DIR  = Path("data/v7_upload")
OUTPUT_DIR = Path("data/to_annotate_minority")
CONF       = 0.20
DEVICE     = "cpu"
IMGSZ      = 640

MINORITY = {0:"Motorcycle", 3:"bicycle", 4:"bus", 7:"truck"}
TARGETS  = {0:80, 3:80, 4:80, 7:80}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for name in MINORITY.values():
    (OUTPUT_DIR / name).mkdir(exist_ok=True)

all_images = sorted(INPUT_DIR.glob("*.jpg"))
print(f"Images to scan: {len(all_images)}")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
model = YOLO(MODEL_PATH)

counts = defaultdict(int)
total  = 0

for i, img_path in enumerate(all_images):
    if all(counts[c] >= TARGETS[c] for c in TARGETS):
        print(f"\n✅ All targets met at image {i+1}")
        break
    try:
        results = model.predict(source=str(img_path), conf=CONF,
                                verbose=False, device=DEVICE, imgsz=IMGSZ)
    except Exception:
        continue

    best = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id in MINORITY:
            if cls_id not in best or conf > best[cls_id]:
                best[cls_id] = conf

    for cls_id in best:
        if counts[cls_id] < TARGETS[cls_id]:
            dest = OUTPUT_DIR / MINORITY[cls_id] / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
                counts[cls_id] += 1
                total += 1

    if (i+1) % 200 == 0:
        print(f"[{i+1:4d}/2953] Motorcycle:{counts[0]} Bicycle:{counts[3]} Bus:{counts[4]} Truck:{counts[7]} Total:{total}")

print(f"\nDONE — {total} images selected")
for cls_id, name in MINORITY.items():
    print(f"  {name:<12}: {counts[cls_id]}/80")
print(f"\nSaved to: {OUTPUT_DIR}/")