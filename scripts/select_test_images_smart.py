"""
select_test_images_smart.py — Select 100 HIGH-QUALITY test images
==================================================================
Filters OUT:
  - Dark/night images with no visible content (low pixel variance)
  - Blank or broken images (tiny file size)
  - Overexposed/washed out images

Keeps images that actually have vehicles, people, and objects visible.

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/select_test_images_smart.py
"""

import os
import re
import random
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
UO_DATASET_DIR = "data/uo_dataset_2"
TRAINING_DIR = "data/ground_truth_roboflow"
OUTPUT_DIR = "data/test_set_for_annotation"
TARGET_COUNT = 100
MIN_CAMERAS = 15

# Quality filters
MIN_FILE_SIZE_KB = 30          # Skip images under 30KB (likely blank/dark)
MIN_BRIGHTNESS = 25            # Skip very dark images (0-255 scale)
MAX_BRIGHTNESS = 240           # Skip overexposed images
MIN_VARIANCE = 400             # Skip low-detail images (flat/blank)

random.seed(42)


# ============================================================
# EXCLUSION SET
# ============================================================
def get_training_stems(training_dir):
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
            if ".rf." in name:
                name = name.split(".rf.")[0]
            if name.endswith("_jpg"):
                name = name[:-4]
            stems.add(name.lower())
    return stems


def is_excluded(camera, filename_stem, excluded_stems):
    combo1 = f"{camera}_{filename_stem}".lower()
    combo2 = f"{camera}__{filename_stem}".lower()
    for exc in excluded_stems:
        if exc.startswith(combo1) or exc.startswith(combo2):
            return True
        date_part = filename_stem[:15]
        if camera.lower() in exc and date_part.lower() in exc:
            return True
    return False


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================
def check_image_quality(img_path):
    """
    Returns (is_good, brightness, variance, reason)
    Checks file size, brightness, and pixel variance.
    """
    # Check file size first (fast)
    size_kb = img_path.stat().st_size / 1024
    if size_kb < MIN_FILE_SIZE_KB:
        return False, 0, 0, f"too small ({size_kb:.0f}KB)"

    try:
        img = Image.open(img_path).convert("L")  # grayscale
        pixels = np.array(img, dtype=np.float32)

        brightness = np.mean(pixels)
        variance = np.var(pixels)

        if brightness < MIN_BRIGHTNESS:
            return False, brightness, variance, f"too dark (brightness={brightness:.0f})"

        if brightness > MAX_BRIGHTNESS:
            return False, brightness, variance, f"overexposed (brightness={brightness:.0f})"

        if variance < MIN_VARIANCE:
            return False, brightness, variance, f"low detail (variance={variance:.0f})"

        return True, brightness, variance, "good"

    except Exception as e:
        return False, 0, 0, f"error: {e}"


# ============================================================
# PARSE HOUR
# ============================================================
def parse_hour(filename):
    stem = Path(filename).stem
    match = re.match(r'(\d{8})_(\d{6})', stem)
    if not match:
        return None
    hour = int(match.group(2)[:2])
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
    print("  SMART TEST SET SELECTOR - quality filtered")
    print("=" * 60)

    excluded = get_training_stems(TRAINING_DIR)
    print(f"\n1. Excluding {len(excluded)} training images")

    print(f"\n2. Scanning {UO_DATASET_DIR} with quality checks...")
    uo_path = Path(UO_DATASET_DIR)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    candidates = defaultdict(list)
    all_cameras = set()
    total = 0
    skipped_dup = 0
    skipped_quality = 0
    quality_reasons = defaultdict(int)

    for camera_dir in sorted(uo_path.iterdir()):
        if not camera_dir.is_dir():
            continue
        camera_name = camera_dir.name
        cam_checked = 0
        cam_good = 0

        for img_path in sorted(camera_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            total += 1

            hour = parse_hour(img_path.name)
            if hour is None:
                continue

            if is_excluded(camera_name, img_path.stem, excluded):
                skipped_dup += 1
                continue

            # Quality check
            is_good, brightness, variance, reason = check_image_quality(img_path)
            cam_checked += 1

            if not is_good:
                skipped_quality += 1
                quality_reasons[reason.split("(")[0].strip()] += 1
                continue

            cam_good += 1
            bucket = get_time_bucket(hour)
            all_cameras.add(camera_name)
            candidates[bucket].append({
                "path": img_path,
                "camera": camera_name,
                "hour": hour,
                "bucket": bucket,
                "brightness": brightness,
                "variance": variance,
                "size_kb": img_path.stat().st_size / 1024,
            })

        if cam_checked > 0:
            print(f"   {camera_name[:40]:40s}  {cam_good}/{cam_checked} passed quality")

    print(f"\n   Total scanned: {total}")
    print(f"   Training duplicates: {skipped_dup}")
    print(f"   Failed quality: {skipped_quality}")
    print(f"     Reasons:")
    for reason, count in sorted(quality_reasons.items(), key=lambda x: -x[1]):
        print(f"       {reason}: {count}")
    print(f"   Good candidates: {sum(len(v) for v in candidates.values())}")
    print(f"   Unique cameras: {len(all_cameras)}")
    print(f"   Per bucket:")
    for b in ["morning", "daytime", "evening", "night"]:
        items = candidates.get(b, [])
        cams = len(set(i["camera"] for i in items))
        print(f"     {b:10s}: {len(items):5d} images, {cams} cameras")

    # Select: prefer HIGH variance images (more content/objects)
    print(f"\n3. Selecting {TARGET_COUNT} best images...")
    IMAGES_PER_BUCKET = TARGET_COUNT // 4

    selected = []
    selected_cameras = set()

    for bucket in ["daytime", "morning", "evening", "night"]:
        pool = candidates.get(bucket, [])
        target_n = IMAGES_PER_BUCKET

        if not pool:
            print(f"   WARNING: No images in {bucket}!")
            continue

        # Sort by variance (highest first = most detail/objects)
        by_camera = defaultdict(list)
        for item in pool:
            by_camera[item["camera"]].append(item)

        # Sort each camera's images by variance (best first)
        for cam in by_camera:
            by_camera[cam].sort(key=lambda x: x["variance"], reverse=True)

        bucket_selected = []
        camera_list = list(by_camera.keys())
        random.shuffle(camera_list)

        # First pass: pick BEST image per camera
        for cam in camera_list:
            if len(bucket_selected) >= target_n:
                break
            img = by_camera[cam][0]  # highest variance = most content
            bucket_selected.append(img)
            selected_cameras.add(cam)

        # Second pass: fill with next-best from any camera
        if len(bucket_selected) < target_n:
            remaining = []
            for cam in camera_list:
                remaining.extend(by_camera[cam][1:])
            remaining.sort(key=lambda x: x["variance"], reverse=True)
            for item in remaining[:target_n - len(bucket_selected)]:
                bucket_selected.append(item)
                selected_cameras.add(item["camera"])

        selected.extend(bucket_selected)
        cams_n = len(set(s["camera"] for s in bucket_selected))
        avg_var = np.mean([s["variance"] for s in bucket_selected])
        print(f"   {bucket:10s}: {len(bucket_selected)}/{target_n} from {cams_n} cameras (avg variance: {avg_var:.0f})")

    print(f"\n   TOTAL: {len(selected)} images from {len(selected_cameras)} cameras")

    # Copy
    print(f"\n4. Copying to {OUTPUT_DIR}...")
    out_path = Path(OUTPUT_DIR)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    manifest = out_path / "_manifest.csv"
    with open(manifest, "w") as f:
        f.write("filename,camera,hour,time_bucket,brightness,variance,size_kb,original_path\n")
        for item in sorted(selected, key=lambda x: (x["bucket"], x["camera"])):
            src = item["path"]
            dst_name = f"{item['camera']}__{src.name}"
            dst = out_path / dst_name
            shutil.copy2(src, dst)
            f.write(f"{dst_name},{item['camera']},{item['hour']},{item['bucket']},"
                    f"{item['brightness']:.1f},{item['variance']:.0f},{item['size_kb']:.0f},{src}\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE! {len(selected)} quality images -> {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    print(f"\n  Time distribution:")
    for b in ["morning", "daytime", "evening", "night"]:
        count = sum(1 for s in selected if s["bucket"] == b)
        hours = sorted(set(s["hour"] for s in selected if s["bucket"] == b))
        avg_b = np.mean([s["brightness"] for s in selected if s["bucket"] == b]) if count else 0
        print(f"    {b:10s}: {count:3d} images  hours={hours}  avg_brightness={avg_b:.0f}")

    print(f"\n  Quality stats:")
    print(f"    Avg brightness: {np.mean([s['brightness'] for s in selected]):.0f}")
    print(f"    Avg variance:   {np.mean([s['variance'] for s in selected]):.0f}")
    print(f"    Avg file size:  {np.mean([s['size_kb'] for s in selected]):.0f}KB")
    print(f"    Min file size:  {min(s['size_kb'] for s in selected):.0f}KB")

    print(f"\n  NEXT: Upload {OUTPUT_DIR}/ to Roboflow and annotate")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()