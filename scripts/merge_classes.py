"""
Merge 8 classes → 5 classes for retraining
============================================
OLD 8 classes          NEW 5 classes
─────────────────────────────────────────
0  Motorcycle      →   3  cyclist
1  Roadwork        →   DROP
2  Traffic light   →   2  traffic_light
3  bicycle         →   3  cyclist
4  bus             →   1  large_vehicle
5  car             →   0  car
6  person          →   4  person
7  truck           →   1  large_vehicle

Creates a NEW dataset folder: data/Newcastle-Traffic-Detection.v7_merged/
Original v7 dataset is UNTOUCHED.

Author: Sumit Malviya (W24041293)
"""

import os
import shutil
from pathlib import Path

# ============================================================
# CLASS MAPPING
# ============================================================
# old_class_id → new_class_id  (None = drop this class)
CLASS_MAP = {
    0: 3,     # Motorcycle  → cyclist
    1: None,  # Roadwork    → DROP
    2: 2,     # Traffic light → traffic_light
    3: 3,     # bicycle     → cyclist
    4: 1,     # bus         → large_vehicle
    5: 0,     # car         → car
    6: 4,     # person      → person
    7: 1,     # truck       → large_vehicle
}

NEW_NAMES = ['car', 'large_vehicle', 'traffic_light', 'cyclist', 'person']

# ============================================================
# PATHS
# ============================================================
SRC = Path('data/Newcastle-Traffic-Detection.v7i.yolov8')
DST = Path('data/Newcastle-Traffic-Detection.v7_merged')

# ============================================================
# CONVERT LABELS
# ============================================================
def convert_labels(src_lbl_dir, dst_lbl_dir):
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    dropped = 0
    kept = 0

    for lbl_file in src_lbl_dir.glob('*.txt'):
        new_lines = []
        for line in open(lbl_file).readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(parts[0])
            new_cls = CLASS_MAP.get(old_cls)
            if new_cls is None:
                dropped += 1
                continue
            new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
            kept += 1
            total += 1

        # Write even if empty (background image)
        with open(dst_lbl_dir / lbl_file.name, 'w') as f:
            f.writelines(new_lines)

    return kept, dropped


print("="*60)
print("  CLASS MERGER: 8 → 5 classes")
print("="*60)
print(f"  Source: {SRC}")
print(f"  Dest:   {DST}")
print()
print("  Mapping:")
old_names = {0:'Motorcycle',1:'Roadwork',2:'Traffic light',
             3:'bicycle',4:'bus',5:'car',6:'person',7:'truck'}
for old, new in CLASS_MAP.items():
    new_name = NEW_NAMES[new] if new is not None else 'DROPPED'
    print(f"    {old} {old_names[old]:15s} → {new_name}")
print()

# Process each split
for split in ['train', 'valid', 'test']:
    src_img = SRC / split / 'images'
    src_lbl = SRC / split / 'labels'
    dst_img = DST / split / 'images'
    dst_lbl = DST / split / 'labels'

    if not src_img.exists():
        print(f"  Skipping {split} — not found")
        continue

    # Copy images
    dst_img.mkdir(parents=True, exist_ok=True)
    img_count = 0
    for img in src_img.glob('*.jpg'):
        shutil.copy2(img, dst_img / img.name)
        img_count += 1

    # Convert labels
    kept, dropped = convert_labels(src_lbl, dst_lbl)

    print(f"  {split:6s}: {img_count} images | {kept} labels kept | {dropped} roadwork labels dropped")

# Write new data.yaml
yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(NEW_NAMES)}
names: {NEW_NAMES}
"""
with open(DST / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print()
print("="*60)
print("  DONE")
print("="*60)
print(f"  New dataset: {DST}/")
print(f"  Classes: {NEW_NAMES}")
print()
print("  NEXT — retrain on Colab:")
print("  1. Zip the new dataset:")
print("     zip -r v7_merged.zip data/Newcastle-Traffic-Detection.v7_merged/")
print("  2. Upload to Google Drive")
print("  3. Train with nc=5")
