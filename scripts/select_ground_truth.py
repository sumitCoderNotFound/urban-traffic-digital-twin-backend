"""
select_ground_truth.py - Select 150 Ground Truth Images for Manual Annotation
==============================================================================
Project: Real-Time Traffic State Estimation Using Deep Learning
Author:  Sumit Malviya (W24041293)

WHAT THIS DOES:
- Reads the auto_label_detections.csv to understand each image
- Selects 150 images covering ALL conditions:
  * Busy scenes (many vehicles + pedestrians)
  * Medium scenes (typical traffic)
  * Empty scenes (no detections - tests false positive rate)
  * High confidence detections (model is sure)
  * Low confidence detections (model is struggling)
  * Different cameras (spatial coverage)
- Copies selected images to data/ground_truth/ with camera name in filename
- Creates a checklist CSV for manual annotation tracking

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/select_ground_truth.py

AFTER RUNNING:
    Upload data/ground_truth/ images to Roboflow for manual annotation
"""

import csv
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────
CSV_PATH = "data/results/auto_label_detections.csv"
IMAGE_DIR = "data/uo_dataset"
OUTPUT_DIR = "data/ground_truth"
CHECKLIST_PATH = "data/results/ground_truth_checklist.csv"
TOTAL_IMAGES = 150

# ── Read CSV ─────────────────────────────────────────────────
print("Reading auto-label results...")
rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["total_objects"] = int(row["total_objects"])
        row["vehicles"] = int(row["vehicles"])
        row["pedestrians"] = int(row["pedestrians"])
        row["cyclists"] = int(row["cyclists"])
        row["avg_confidence"] = float(row["avg_confidence"])
        rows.append(row)

print(f"Total images in CSV: {len(rows)}")

# ── Categorise images ────────────────────────────────────────
busy = []          # 5+ total objects (busy scenes)
medium = []        # 1-4 total objects (typical)
empty = []         # 0 total objects (empty/night)
has_pedestrians = []  # images with people
has_cyclists = []     # images with cyclists (rare, valuable)
low_conf = []      # avg confidence < 0.3 (model struggling)
high_conf = []     # avg confidence > 0.5 (model confident)

for row in rows:
    n = row["total_objects"]
    conf = row["avg_confidence"]
    
    if n >= 5:
        busy.append(row)
    elif 1 <= n <= 4:
        medium.append(row)
    else:
        empty.append(row)
    
    if row["pedestrians"] > 0:
        has_pedestrians.append(row)
    if row["cyclists"] > 0:
        has_cyclists.append(row)
    if conf > 0 and conf < 0.3:
        low_conf.append(row)
    if conf > 0.5:
        high_conf.append(row)

print(f"\nImage categories:")
print(f"  Busy (5+ objects):     {len(busy)}")
print(f"  Medium (1-4 objects):  {len(medium)}")
print(f"  Empty (0 objects):     {len(empty)}")
print(f"  Has pedestrians:       {len(has_pedestrians)}")
print(f"  Has cyclists:          {len(has_cyclists)}")
print(f"  Low confidence (<0.3): {len(low_conf)}")
print(f"  High confidence (>0.5):{len(high_conf)}")

# ── Selection strategy ───────────────────────────────────────
# Goal: 150 images covering all conditions and cameras
selected = {}  # key = "camera__image" to avoid duplicates

def add_images(source_list, count, label):
    """Pick random images from a category, spread across cameras."""
    random.seed(42)  # Reproducible selection
    
    # Group by camera
    by_camera = defaultdict(list)
    for row in source_list:
        key = f"{row['camera']}__{row['image']}"
        if key not in selected:
            by_camera[row["camera"]].append(row)
    
    # Pick evenly across cameras
    added = 0
    cameras = list(by_camera.keys())
    random.shuffle(cameras)
    
    while added < count and cameras:
        for cam in list(cameras):
            if added >= count:
                break
            if by_camera[cam]:
                row = random.choice(by_camera[cam])
                by_camera[cam].remove(row)
                key = f"{row['camera']}__{row['image']}"
                selected[key] = {**row, "category": label}
                added += 1
            else:
                cameras.remove(cam)
    
    print(f"  Selected {added} {label} images")
    return added

print(f"\nSelecting {TOTAL_IMAGES} ground truth images...")

# ALL cyclists first (they are rare and valuable for research)
cyclists_to_add = min(len(has_cyclists), 10)
for row in has_cyclists[:cyclists_to_add]:
    key = f"{row['camera']}__{row['image']}"
    selected[key] = {**row, "category": "has_cyclist"}
print(f"  Selected {cyclists_to_add} cyclist images (all available)")

# Now fill remaining slots
remaining = TOTAL_IMAGES - len(selected)

add_images(busy, 40, "busy")
add_images(medium, 35, "medium")
add_images(empty, 20, "empty")
add_images(has_pedestrians, 20, "has_pedestrian")
add_images(low_conf, 15, "low_confidence")
add_images(high_conf, 10, "high_confidence")

# If still short, add random from any category
if len(selected) < TOTAL_IMAGES:
    remaining = TOTAL_IMAGES - len(selected)
    add_images(rows, remaining, "random_fill")

print(f"\nTotal selected: {len(selected)}")

# ── Copy images to ground_truth folder ───────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

copied = 0
checklist_rows = []

for key, info in sorted(selected.items()):
    camera = info["camera"]
    image_name = info["image"]
    
    # Find source image
    src_path = Path(IMAGE_DIR) / camera / image_name
    if not src_path.exists():
        # Try without camera subfolder
        src_path = Path(IMAGE_DIR) / image_name
        if not src_path.exists():
            print(f"  WARNING: Not found: {src_path}")
            continue
    
    # Copy with camera name in filename for easy identification
    dst_name = f"{camera}__{image_name}"
    dst_path = Path(OUTPUT_DIR) / dst_name
    shutil.copy2(src_path, dst_path)
    copied += 1
    
    # Build checklist row
    checklist_rows.append({
        "filename": dst_name,
        "camera": camera,
        "original_image": image_name,
        "category": info["category"],
        "auto_vehicles": info["vehicles"],
        "auto_pedestrians": info["pedestrians"],
        "auto_cyclists": info["cyclists"],
        "auto_total": info["total_objects"],
        "auto_confidence": info["avg_confidence"],
        "manual_vehicles": "",       # YOU FILL THIS IN
        "manual_pedestrians": "",    # YOU FILL THIS IN
        "manual_cyclists": "",       # YOU FILL THIS IN
        "manual_total": "",          # YOU FILL THIS IN
        "annotated": "NO",           # Change to YES after annotating
        "notes": "",                 # Any observations
    })

# ── Save checklist CSV ───────────────────────────────────────
with open(CHECKLIST_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=checklist_rows[0].keys())
    writer.writeheader()
    writer.writerows(checklist_rows)

# ── Summary ──────────────────────────────────────────────────
categories = defaultdict(int)
cameras_covered = set()
for info in selected.values():
    categories[info["category"]] += 1
    cameras_covered.add(info["camera"])

print(f"\n{'='*60}")
print(f"  GROUND TRUTH SELECTION COMPLETE")
print(f"{'='*60}")
print(f"  Images copied:    {copied}")
print(f"  Cameras covered:  {len(cameras_covered)}")
print(f"{'─'*60}")
print(f"  BREAKDOWN BY CATEGORY:")
for cat, count in sorted(categories.items()):
    print(f"    {cat:25s}  {count}")
print(f"{'─'*60}")
print(f"  Images saved to:  {OUTPUT_DIR}/")
print(f"  Checklist CSV:    {CHECKLIST_PATH}")
print(f"{'='*60}")
print(f"\n  NEXT STEPS:")
print(f"  1. Open {OUTPUT_DIR}/ and visually check the images")
print(f"  2. Upload to Roboflow for manual annotation")
print(f"  3. For each image, draw boxes around ALL objects")
print(f"  4. Update the checklist CSV with manual counts")
print(f"  5. Use for RQ1 baseline evaluation")
print(f"{'='*60}")