"""
Extract 1500 minority-class images for v7 training upload
==========================================================
Scans all available image folders, runs YOLOv8 to find images
containing minority classes (bus, truck, motorcycle, bicycle, roadwork),
then selects the best 1500 for Roboflow upload.

Priority order:
  1. motorcycle  (rarest — 0% baseline AP)
  2. bicycle     (rare — 0% improvement after fine-tune)
  3. roadwork    (domain-specific — not in COCO)
  4. bus         (confused with truck)
  5. truck       (confused with bus)

Output: data/v7_upload/  — 1500 images ready for Roboflow

Author: Sumit Malviya (W24041293)
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
TARGET      = 1500
CONF        = 0.25
DEVICE      = "cpu"
IMGSZ       = 640   # faster screening — use 640 not 960

OUTPUT_DIR  = Path("data/v7_upload")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source folders to scan (in priority order)
SOURCES = [
    Path("data/minority_class_collection"),   # already targeted minority
    Path("data/minority_class_images"),        # extra minority
    Path("data/uo_dataset_midday"),            # Friday midday — good lighting
    Path("data/dataset"),                      # scheduler collected
    Path("data/images"),                       # scheduler collected
]

# COCO class IDs for minority classes
MINORITY_COCO = {
    1:  "bicycle",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Priority weights — higher = more important to include
PRIORITY = {
    "motorcycle": 5,
    "bicycle":    4,
    "roadwork":   4,
    "bus":        3,
    "truck":      2,
}

# ============================================================
# COLLECT ALL IMAGE PATHS
# ============================================================
print(f"\n{'='*60}")
print(f"  MINORITY CLASS IMAGE EXTRACTOR")
print(f"  Target: {TARGET} images for v7 training")
print(f"{'='*60}\n")

all_images = []
seen_hashes = set()

for source in SOURCES:
    if not source.exists():
        print(f"  Skipping {source} (not found)")
        continue
    imgs = list(source.rglob("*.jpg")) + list(source.rglob("*.jpeg")) + list(source.rglob("*.png"))
    print(f"  {source}: {len(imgs)} images")
    all_images.extend(imgs)

print(f"\n  Total images to scan: {len(all_images)}")
print(f"  Loading YOLOv8n for screening...\n")

model = YOLO("yolov8n.pt")   # use nano for fast screening

# ============================================================
# SCAN AND SCORE EACH IMAGE
# ============================================================
scored = []
checked = 0
skipped_dupes = 0

for img_path in all_images:
    # Dedup by file hash
    try:
        with open(img_path, "rb") as f:
            h = hashlib.md5(f.read(8192)).hexdigest()  # partial hash for speed
        if h in seen_hashes:
            skipped_dupes += 1
            continue
        seen_hashes.add(h)
    except Exception:
        continue

    # Run detection
    try:
        results = model.predict(str(img_path), conf=CONF,
                                verbose=False, device=DEVICE, imgsz=IMGSZ)
    except Exception:
        continue

    # Score the image
    score = 0
    classes_found = defaultdict(int)

    for b in results[0].boxes:
        cid  = int(b.cls[0])
        conf = float(b.conf[0])
        if cid in MINORITY_COCO:
            cls_name = MINORITY_COCO[cid]
            classes_found[cls_name] += 1
            score += PRIORITY.get(cls_name, 1) * conf

    if score > 0:
        scored.append({
            "path":    img_path,
            "score":   score,
            "classes": dict(classes_found),
        })

    checked += 1
    if checked % 500 == 0:
        print(f"  Checked {checked}/{len(all_images)} | "
              f"Found {len(scored)} minority | Dupes skipped: {skipped_dupes}")

print(f"\n  Scan complete: {checked} images checked")
print(f"  Duplicates skipped: {skipped_dupes}")
print(f"  Images with minority classes: {len(scored)}")

# ============================================================
# SELECT TOP 1500 BY SCORE
# ============================================================
# Sort by score descending — highest priority classes first
scored.sort(key=lambda x: x["score"], reverse=True)

selected = scored[:TARGET]
actual   = len(selected)

print(f"\n  Selecting top {actual} images by minority class score...")

# Class distribution in selection
class_dist = defaultdict(int)
for item in selected:
    for cls, cnt in item["classes"].items():
        class_dist[cls] += cnt

print(f"\n  Class distribution in selected {actual} images:")
for cls in ["motorcycle", "bicycle", "roadwork", "bus", "truck"]:
    count = class_dist.get(cls, 0)
    bar   = "█" * min(count // 5, 40)
    print(f"    {cls:12s}: {count:4d}  {bar}")

# ============================================================
# COPY TO OUTPUT FOLDER
# ============================================================
print(f"\n  Copying {actual} images to {OUTPUT_DIR}/...")

copied = 0
for item in selected:
    src  = item["path"]
    # Clean filename: keep camera name + original stem
    stem = src.stem[:60].replace(" ", "_")
    dest = OUTPUT_DIR / f"{stem}_{copied:04d}.jpg"
    try:
        shutil.copy2(src, dest)
        copied += 1
    except Exception as e:
        print(f"  Warning: could not copy {src.name}: {e}")

    if copied % 200 == 0:
        print(f"  Copied {copied}/{actual}...")

# ============================================================
# SUMMARY
# ============================================================
final_count = len(list(OUTPUT_DIR.glob("*.jpg")))

print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
print(f"  Images copied to: {OUTPUT_DIR}/")
print(f"  Total images:     {final_count}")
print(f"  Existing train:   789  (v6)")
print(f"  New images:       {final_count}")
print(f"  Expected v7 total: {789 + final_count} training images")
print(f"\n  NEXT STEPS:")
print(f"  1. Upload data/v7_upload/ to Roboflow")
print(f"  2. Set split: Train 100% (do NOT add to test)")
print(f"  3. Auto-label using your existing v6 model")
print(f"  4. Review + fix mistakes (focus on motorcycle, bicycle, roadwork)")
print(f"  5. Export as v7 → YOLOv8 format")
print(f"  6. Retrain: python scripts/rq2_improved_training.py --data data/Newcastle-Traffic-Detection.v7i.yolov8")
print(f"{'='*60}")
