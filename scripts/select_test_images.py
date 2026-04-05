"""
select_test_images.py — Select 100 stratified test images for annotation
=========================================================================
Project: Real-Time Traffic State Estimation Using Deep Learning
Author:  Sumit Malviya (W24041293)

Structure: data/uo_dataset_2/CameraName/YYYYMMDD_HHMMSS.jpg

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/select_test_images.py
"""

import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
UO_DATASET_DIR = "data/uo_dataset_2"
TRAINING_DIR = "data/ground_truth_roboflow"
OUTPUT_DIR = "data/test_set_for_annotation"
TARGET_COUNT = 100
MIN_CAMERAS = 15

random.seed(42)


# ============================================================
# STEP 1: Build exclusion set from training images
# ============================================================
def get_training_stems(training_dir):
    """Extract original camera+timestamp from Roboflow filenames."""
    stems = set()
    train_path = Path(training_dir)

    for split in ["train", "valid", "test"]:
        img_dir = train_path / split / "images"
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            name = img.stem

            # Remove Roboflow hash
            if ".rf." in name:
                name = name.split(".rf.")[0]
            # Remove _jpg suffix Roboflow adds
            if name.endswith("_jpg"):
                name = name[:-4]

            stems.add(name.lower())

    return stems


def is_excluded(camera, filename_stem, excluded_stems):
    """Check if camera+filename combo matches any training image."""
    combo1 = f"{camera}_{filename_stem}".lower()
    combo2 = f"{camera}__{filename_stem}".lower()

    for exc in excluded_stems:
        if exc.startswith(combo1) or exc.startswith(combo2):
            return True
        # Also check date match
        date_part = filename_stem[:15]  # YYYYMMDD_HHMMSS
        if camera.lower() in exc and date_part.lower() in exc:
            return True

    return False


# ============================================================
# STEP 2: Parse hour from filename
# ============================================================
def parse_hour(filename):
    """
    Extract hour from filenames like:
      20260324_163844.jpg       -> hour 16
      20260324_165056_0c2f0d.jpg -> hour 16
    """
    stem = Path(filename).stem
    match = re.match(r'(\d{8})_(\d{6})', stem)
    if not match:
        return None
    time_str = match.group(2)
    hour = int(time_str[:2])
    return hour if 0 <= hour <= 23 else None


def get_time_bucket(hour):
    if 7 <= hour < 10:
        return "morning"
    elif 10 <= hour < 16:
        return "daytime"
    elif 16 <= hour < 20:
        return "evening"
    else:
        return "night"


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  TEST SET SELECTOR — 100 stratified images")
    print("=" * 60)

    # Step 1: Exclusion set
    print(f"\n1. Building exclusion set from {TRAINING_DIR}...")
    excluded_stems = get_training_stems(TRAINING_DIR)
    print(f"   Excluding {len(excluded_stems)} training images")

    # Step 2: Scan uo_dataset (subfolders = cameras)
    print(f"\n2. Scanning {UO_DATASET_DIR}...")
    uo_path = Path(UO_DATASET_DIR)
    if not uo_path.exists():
        print(f"   ERROR: {UO_DATASET_DIR} not found!")
        return

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    candidates = defaultdict(list)
    all_cameras = set()
    total_scanned = 0
    skipped_dup = 0
    skipped_parse = 0

    for camera_dir in sorted(uo_path.iterdir()):
        if not camera_dir.is_dir():
            continue

        camera_name = camera_dir.name

        for img_path in sorted(camera_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            total_scanned += 1
            hour = parse_hour(img_path.name)

            if hour is None:
                skipped_parse += 1
                continue

            if is_excluded(camera_name, img_path.stem, excluded_stems):
                skipped_dup += 1
                continue

            bucket = get_time_bucket(hour)
            all_cameras.add(camera_name)
            candidates[bucket].append({
                "path": img_path,
                "camera": camera_name,
                "hour": hour,
                "bucket": bucket,
            })

    print(f"   Total images scanned: {total_scanned}")
    print(f"   Duplicates excluded: {skipped_dup}")
    print(f"   Unparseable: {skipped_parse}")
    print(f"   Valid candidates: {sum(len(v) for v in candidates.values())}")
    print(f"   Unique cameras: {len(all_cameras)}")
    print(f"   Per bucket:")
    for bucket in ["morning", "daytime", "evening", "night"]:
        items = candidates.get(bucket, [])
        cams = len(set(item["camera"] for item in items))
        print(f"     {bucket:10s}: {len(items):5d} images from {cams} cameras")

    # Step 3: Stratified selection
    print(f"\n3. Selecting {TARGET_COUNT} images...")
    IMAGES_PER_BUCKET = TARGET_COUNT // 4

    selected = []
    selected_cameras = set()

    for bucket in ["daytime", "morning", "evening", "night"]:
        pool = candidates.get(bucket, [])
        target_n = IMAGES_PER_BUCKET

        if not pool:
            print(f"   WARNING: No images in {bucket} bucket!")
            continue

        by_camera = defaultdict(list)
        for item in pool:
            by_camera[item["camera"]].append(item)

        bucket_selected = []
        camera_list = list(by_camera.keys())
        random.shuffle(camera_list)

        # First pass: 1 per camera for diversity
        for cam in camera_list:
            if len(bucket_selected) >= target_n:
                break
            img = random.choice(by_camera[cam])
            bucket_selected.append(img)
            selected_cameras.add(cam)

        # Second pass: fill remaining
        if len(bucket_selected) < target_n:
            remaining = [i for i in pool if i not in bucket_selected]
            random.shuffle(remaining)
            for item in remaining[:target_n - len(bucket_selected)]:
                bucket_selected.append(item)
                selected_cameras.add(item["camera"])

        selected.extend(bucket_selected)
        cams_in_bucket = len(set(s["camera"] for s in bucket_selected))
        print(f"   {bucket:10s}: {len(bucket_selected)}/{target_n} from {cams_in_bucket} cameras")

    # Fill if short
    while len(selected) < TARGET_COUNT:
        largest = max(candidates.keys(), key=lambda b: len(candidates[b]))
        remaining = [i for i in candidates[largest] if i not in selected]
        if not remaining:
            break
        selected.append(random.choice(remaining))

    print(f"\n   TOTAL: {len(selected)} images from {len(selected_cameras)} cameras")

    # Step 4: Copy
    print(f"\n4. Copying to {OUTPUT_DIR}...")
    out_path = Path(OUTPUT_DIR)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    manifest_path = out_path / "_manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("filename,camera,hour,time_bucket,original_path\n")
        for item in sorted(selected, key=lambda x: (x["bucket"], x["camera"])):
            src = item["path"]
            dst_name = f"{item['camera']}__{src.name}"
            dst = out_path / dst_name
            shutil.copy2(src, dst)
            f.write(f"{dst_name},{item['camera']},{item['hour']},{item['bucket']},{src}\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE! {len(selected)} images -> {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    print(f"\n  Time distribution:")
    for bucket in ["morning", "daytime", "evening", "night"]:
        count = sum(1 for s in selected if s["bucket"] == bucket)
        hours = sorted(set(s["hour"] for s in selected if s["bucket"] == bucket))
        print(f"    {bucket:10s}: {count:3d} images  (hours: {hours})")

    print(f"\n  Top cameras:")
    cam_counts = defaultdict(int)
    for s in selected:
        cam_counts[s["camera"]] += 1
    for cam, count in sorted(cam_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {cam[:45]:45s} {count:3d}")

    print(f"\n  Manifest: {manifest_path}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Upload {OUTPUT_DIR}/ folder to Roboflow")
    print(f"  2. Annotate with: Traffic light, Bus, Car, Person, Truck")
    print(f"  3. Export as YOLOv8 format")
    print(f"  4. Use ONLY as test split — never train on these")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()