"""
collect_minority_classes.py - Targeted collection for weak classes
===================================================================
Collects LIVE from Newcastle Urban Observatory every 10 minutes
until 5pm. Only saves images where YOLO detects minority classes:
bus, truck, person, bicycle, motorcycle.

Focuses on city center cameras where these classes are most common.

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/collect_minority_classes.py

Author: Sumit Malviya (W24041293)
"""

import asyncio
import hashlib
import shutil
import time
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
END_HOUR = 17  # Stop at 5pm
INTERVAL = 600  # 10 minutes between runs (more frequent = more unique frames)

# Where to save
SAVE_DIR = Path("data/minority_class_collection")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Separate folders for easy upload
(SAVE_DIR / "for_training").mkdir(exist_ok=True)
(SAVE_DIR / "for_test").mkdir(exist_ok=True)

# COCO class IDs we want
MINORITY_CLASSES = {
    0: "person",
    1: "bicycle",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# City center cameras (high pedestrian/bus/cyclist activity)
PRIORITY_CAMERAS = [
    "newcastle", "gateshead", "sunderland", "durham",
    "darlington", "north_tyneside", "south_tyneside",
    "high_bondgate", "new_elvet", "bus_station",
    "stockton", "cumberland", "millburngate",
    "framwellgate", "gilesgate", "crossgate",
    "elvet", "sadler", "market", "station",
]

# Track what we've found
stats = defaultdict(int)
seen_hashes = set()


def is_priority_camera(camera_name):
    """Check if camera is in a city center / busy area."""
    name_lower = camera_name.lower()
    return any(kw in name_lower for kw in PRIORITY_CAMERAS)


async def run_collection(run_num):
    """Single collection run - fetch, detect, save minority class images."""
    from app.services.urban_observatory import UrbanObservatoryCollector
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    collector = UrbanObservatoryCollector()

    now = datetime.now()
    print(f"\n{'=' * 60}")
    print(f"  RUN #{run_num}  |  {now.strftime('%H:%M')}  |  Target: bus, truck, person, bicycle, motorcycle")
    print(f"{'=' * 60}")

    cameras = await collector.get_camera_sensors()
    if not cameras:
        print("  No cameras returned")
        return

    # Sort: priority cameras first
    cameras.sort(key=lambda c: (0 if is_priority_camera(c.get("name", "")) else 1))

    saved_this_run = 0
    found_this_run = defaultdict(int)

    for cam in cameras:
        cam_name = cam.get("name", "unknown")
        image_url = cam.get("image_url", "")
        cam_id = cam.get("camera_id", "unknown")

        # Download image
        img_path = await collector.fetch_camera_image(image_url, cam_id)
        if not img_path:
            continue

        # Check for duplicate (same image content)
        with open(img_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)

        # Run YOLO detection
        results = model.predict(img_path, conf=0.25, verbose=False, device="cpu")

        # Check which minority classes are present
        classes_found = {}
        for box in results[0].boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            if cid in MINORITY_CLASSES:
                cls_name = MINORITY_CLASSES[cid]
                if cls_name not in classes_found or conf > classes_found[cls_name]:
                    classes_found[cls_name] = conf

        if not classes_found:
            continue

        # Determine if this should be training or test
        # Every 5th minority image goes to test, rest to training
        total_saved = sum(stats.values())
        dest_folder = "for_test" if total_saved % 5 == 0 else "for_training"

        # Save with descriptive filename
        clean_name = re.sub(r"[^\w\s\-]", "", cam_name)
        clean_name = re.sub(r"\s+", "_", clean_name.strip())[:40]
        ts = now.strftime("%Y%m%d_%H%M%S")
        classes_str = "_".join(sorted(classes_found.keys()))
        filename = f"{clean_name}__{ts}_{classes_str}_{img_hash[:6]}.jpg"

        dst = SAVE_DIR / dest_folder / filename
        shutil.copy2(img_path, dst)
        saved_this_run += 1

        for cls_name in classes_found:
            found_this_run[cls_name] += 1
            stats[cls_name] += 1

        # Show what we found
        cls_info = ", ".join(f"{k}({v:.0%})" for k, v in classes_found.items())
        priority = "*" if is_priority_camera(cam_name) else " "
        print(f"  {priority} {cam_name[:35]:35s} -> {cls_info:30s} [{dest_folder}]")

    print(f"\n  Run #{run_num} summary: {saved_this_run} images saved")
    print(f"  This run: {dict(found_this_run)}")
    print(f"  TOTAL so far:")
    for cls_name in ["person", "bus", "truck", "bicycle", "motorcycle"]:
        count = stats.get(cls_name, 0)
        bar = "#" * min(count, 50)
        print(f"    {cls_name:12s}: {count:4d} {bar}")

    train_count = len(list((SAVE_DIR / "for_training").glob("*.jpg")))
    test_count = len(list((SAVE_DIR / "for_test").glob("*.jpg")))
    print(f"\n  Training images: {train_count}")
    print(f"  Test images:     {test_count}")


async def main():
    print("=" * 60)
    print("  MINORITY CLASS COLLECTOR")
    print(f"  Running until {END_HOUR}:00")
    print(f"  Interval: {INTERVAL // 60} minutes")
    print(f"  Saving to: {SAVE_DIR}/")
    print(f"  Target classes: {list(MINORITY_CLASSES.values())}")
    print("=" * 60)

    run_num = 0

    while True:
        now = datetime.now()
        if now.hour >= END_HOUR:
            break

        run_num += 1
        try:
            await run_collection(run_num)
        except Exception as e:
            print(f"  ERROR: {e}")

        now = datetime.now()
        if now.hour >= END_HOUR:
            break

        runs_left = ((END_HOUR * 60 - now.hour * 60 - now.minute) // (INTERVAL // 60))
        print(f"\n  Next run in {INTERVAL // 60} min (~{runs_left} runs left until {END_HOUR}:00)")
        await asyncio.sleep(INTERVAL)

    # Final summary
    train_count = len(list((SAVE_DIR / "for_training").glob("*.jpg")))
    test_count = len(list((SAVE_DIR / "for_test").glob("*.jpg")))

    print(f"\n{'=' * 60}")
    print(f"  COLLECTION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total minority class images:")
    for cls_name in ["person", "bus", "truck", "bicycle", "motorcycle"]:
        print(f"    {cls_name:12s}: {stats.get(cls_name, 0)}")
    print(f"\n  For training: {train_count} images -> {SAVE_DIR}/for_training/")
    print(f"  For test:     {test_count} images -> {SAVE_DIR}/for_test/")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Upload {SAVE_DIR}/for_training/ to Roboflow (Train 100%)")
    print(f"  2. Upload {SAVE_DIR}/for_test/ to Roboflow (Test 100%)")
    print(f"  3. Annotate all images with 8 classes")
    print(f"  4. Generate new version and download")
    print(f"  5. Retrain with balanced data")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())