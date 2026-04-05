"""
final_data_collection.py — One-shot collection for research evaluation
========================================================================
Project: Real-Time Traffic State Estimation Using Deep Learning
Author:  Sumit Malviya (W24041293)

WHAT THIS DOES:
  Phase 1: Collects images from ALL Urban Observatory cameras every 15 min
           until END_HOUR (default 18:00). Saves to data/uo_dataset/CameraName/
  Phase 2: After collection ends, automatically selects 100 stratified test
           images (25 morning, 25 daytime, 25 evening, 25 night)
  Phase 3: Generates a collection report CSV with stats for your paper

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/final_data_collection.py

    # Or with custom end time (e.g., run until 8pm):
    python scripts/final_data_collection.py --end-hour 20

    # Skip collection, just select test images from existing data:
    python scripts/final_data_collection.py --skip-collection

LEAVE RUNNING: Start now (5am), leave it running. It will stop at 6pm
automatically, select your test images, and print next steps.
"""

import os
import re
import sys
import json
import time
import random
import shutil
import asyncio
import aiohttp
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
UO_DATASET_DIR = "data/uo_dataset_2"
TRAINING_DIR = "data/ground_truth_roboflow"
TEST_OUTPUT_DIR = "data/test_set_for_annotation"
REPORT_DIR = "data/results"

COLLECTION_INTERVAL = 900  # 15 minutes between runs
DEFAULT_END_HOUR = 18      # Stop collecting at 6pm
TARGET_TEST_IMAGES = 100
MIN_CAMERAS = 15

UO_API_URL = "https://portal.cctv.urbanobservatory.ac.uk/latest"
UO_IMAGE_URL = "https://portal.cctv.urbanobservatory.ac.uk/photo"
HEADERS = {"User-Agent": "UrbanDigitalTwin/1.0 (MSc Research - Northumbria Uni)"}

random.seed(42)


# ============================================================
# PHASE 1: DATA COLLECTION
# ============================================================
async def fetch_camera_list():
    """Get list of all cameras from Urban Observatory API."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(UO_API_URL, headers=HEADERS, ssl=False) as resp:
                if resp.status != 200:
                    print(f"   API returned {resp.status}")
                    return []
                data = await resp.json()

                cameras = []
                if isinstance(data, list):
                    for item in data:
                        cam_id = str(item.get("id", item.get("name", "unknown")))
                        name = item.get("name", cam_id)
                        image_url = item.get("image", item.get("photo", ""))
                        if not image_url and cam_id:
                            image_url = f"{UO_IMAGE_URL}/{cam_id}"
                        cameras.append({
                            "camera_id": cam_id,
                            "name": name,
                            "image_url": image_url,
                        })
                elif isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, dict):
                            image_url = val.get("image", val.get("photo", ""))
                            if not image_url:
                                image_url = f"{UO_IMAGE_URL}/{key}"
                            cameras.append({
                                "camera_id": key,
                                "name": val.get("name", key),
                                "image_url": image_url,
                            })

                return cameras
    except Exception as e:
        print(f"   API error: {e}")
        return []


async def download_image(session, camera, save_dir):
    """Download a single camera image."""
    image_url = camera.get("image_url", "")
    cam_name = camera.get("name", camera.get("camera_id", "unknown"))

    # Clean camera name for folder
    clean_name = re.sub(r'[^\w\s\-]', '', cam_name)
    clean_name = re.sub(r'\s+', '_', clean_name.strip())[:60]
    if not clean_name:
        clean_name = "unknown"

    cam_dir = Path(save_dir) / clean_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = cam_dir / filename

    try:
        async with session.get(image_url, headers=HEADERS, ssl=False) as resp:
            if resp.status != 200:
                return None

            content = await resp.read()

            # Skip tiny/invalid images
            if len(content) < 5000:
                return None

            content_type = resp.headers.get('content-type', '')
            if 'image' not in content_type and 'octet' not in content_type:
                return None

            # Skip duplicate images (same hash as last saved)
            import hashlib
            img_hash = hashlib.md5(content).hexdigest()[:12]

            # Check if we already have this exact image
            hash_file = cam_dir / "_last_hash.txt"
            if hash_file.exists():
                last_hash = hash_file.read_text().strip()
                if last_hash == img_hash:
                    return None  # Same image, skip

            hash_file.write_text(img_hash)

            # Add hash to filename to avoid duplicates
            filepath = cam_dir / f"{timestamp}_{img_hash}.jpg"
            with open(filepath, "wb") as f:
                f.write(content)

            return str(filepath)

    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


async def run_collection_cycle(run_number):
    """Single collection cycle — download from all cameras."""
    now = datetime.now()
    print(f"\n{'=' * 60}")
    print(f"  RUN #{run_number}  |  {now.strftime('%H:%M:%S')}  |  {now.strftime('%A %d %B')}")
    print(f"{'=' * 60}")

    cameras = await fetch_camera_list()
    if not cameras:
        print("   No cameras returned from API")
        return 0

    print(f"   {len(cameras)} cameras found, downloading...")

    saved = 0
    timeout = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Process in batches of 10 to be polite to the API
        batch_size = 10
        for i in range(0, len(cameras), batch_size):
            batch = cameras[i:i + batch_size]
            tasks = [download_image(session, cam, UO_DATASET_DIR) for cam in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, str) and result:
                    saved += 1

            # Small delay between batches
            await asyncio.sleep(0.5)

    print(f"   Saved: {saved} new images (skipped duplicates)")
    return saved


async def collection_phase(end_hour):
    """Run collection every 15 min until end_hour."""
    print("\n" + "=" * 60)
    print("  PHASE 1: DATA COLLECTION")
    print(f"  Collecting every 15 min until {end_hour}:00")
    print(f"  Saving to: {UO_DATASET_DIR}/")
    print("=" * 60)

    run_number = 0
    total_saved = 0

    while True:
        now = datetime.now()

        # Check if we should stop
        if now.hour >= end_hour:
            print(f"\n  Reached {end_hour}:00 — stopping collection.")
            break

        run_number += 1
        saved = await run_collection_cycle(run_number)
        total_saved += saved

        # Calculate time until next run
        now = datetime.now()
        if now.hour >= end_hour:
            break

        next_run = now + timedelta(seconds=COLLECTION_INTERVAL)
        wait_secs = (next_run - datetime.now()).total_seconds()

        if wait_secs > 0:
            mins_left = wait_secs / 60
            runs_left = ((datetime.now().replace(hour=end_hour, minute=0, second=0) -
                         datetime.now()).total_seconds()) / COLLECTION_INTERVAL
            print(f"\n   Next run at {next_run.strftime('%H:%M')} "
                  f"(~{int(runs_left)} runs remaining until {end_hour}:00)")
            print(f"   Total new images so far: {total_saved}")
            await asyncio.sleep(wait_secs)

    print(f"\n  Collection complete! {total_saved} new images across {run_number} runs.")
    return total_saved


# ============================================================
# PHASE 2: TEST IMAGE SELECTION (same logic as before, improved)
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


def select_test_images():
    """Select 100 stratified test images."""
    print("\n" + "=" * 60)
    print("  PHASE 2: TEST IMAGE SELECTION")
    print("=" * 60)

    excluded = get_training_stems(TRAINING_DIR)
    print(f"\n   Excluding {len(excluded)} training images")

    uo_path = Path(UO_DATASET_DIR)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    candidates = defaultdict(list)
    all_cameras = set()
    total = 0
    skipped_dup = 0

    for camera_dir in sorted(uo_path.iterdir()):
        if not camera_dir.is_dir():
            continue
        camera_name = camera_dir.name
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
            bucket = get_time_bucket(hour)
            all_cameras.add(camera_name)
            candidates[bucket].append({
                "path": img_path,
                "camera": camera_name,
                "hour": hour,
                "bucket": bucket,
            })

    print(f"   Scanned: {total} images from {len(all_cameras)} cameras")
    print(f"   Excluded: {skipped_dup} training duplicates")
    print(f"   Available per bucket:")
    for b in ["morning", "daytime", "evening", "night"]:
        items = candidates.get(b, [])
        cams = len(set(i["camera"] for i in items))
        print(f"     {b:10s}: {len(items):5d} images, {cams} cameras")

    # Calculate per-bucket targets — proportional if some buckets are empty
    buckets_with_data = [b for b in ["morning", "daytime", "evening", "night"]
                         if len(candidates.get(b, [])) > 0]
    per_bucket = TARGET_TEST_IMAGES // len(buckets_with_data) if buckets_with_data else 0
    remainder = TARGET_TEST_IMAGES - (per_bucket * len(buckets_with_data))

    print(f"\n   Selecting {TARGET_TEST_IMAGES} images "
          f"({per_bucket} per bucket, {len(buckets_with_data)} buckets with data)")

    selected = []
    selected_cameras = set()

    for idx, bucket in enumerate(buckets_with_data):
        pool = candidates[bucket]
        target_n = per_bucket + (1 if idx < remainder else 0)

        by_camera = defaultdict(list)
        for item in pool:
            by_camera[item["camera"]].append(item)

        bucket_selected = []
        camera_list = list(by_camera.keys())
        random.shuffle(camera_list)

        # First pass: 1 per camera
        for cam in camera_list:
            if len(bucket_selected) >= target_n:
                break
            img = random.choice(by_camera[cam])
            bucket_selected.append(img)
            selected_cameras.add(cam)

        # Second pass: fill
        if len(bucket_selected) < target_n:
            remaining = [i for i in pool if i not in bucket_selected]
            random.shuffle(remaining)
            for item in remaining[:target_n - len(bucket_selected)]:
                bucket_selected.append(item)
                selected_cameras.add(item["camera"])

        selected.extend(bucket_selected)
        cams_n = len(set(s["camera"] for s in bucket_selected))
        print(f"   {bucket:10s}: {len(bucket_selected)}/{target_n} from {cams_n} cameras")

    print(f"\n   TOTAL: {len(selected)} images from {len(selected_cameras)} cameras")

    # Copy to output
    out_path = Path(TEST_OUTPUT_DIR)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    manifest = out_path / "_manifest.csv"
    with open(manifest, "w") as f:
        f.write("filename,camera,hour,time_bucket,original_path\n")
        for item in sorted(selected, key=lambda x: (x["bucket"], x["camera"])):
            src = item["path"]
            dst_name = f"{item['camera']}__{src.name}"
            dst = out_path / dst_name
            shutil.copy2(src, dst)
            f.write(f"{dst_name},{item['camera']},{item['hour']},{item['bucket']},{src}\n")

    return selected, selected_cameras


# ============================================================
# PHASE 3: COLLECTION REPORT
# ============================================================
def generate_report(selected, selected_cameras):
    """Generate a research-ready report of the dataset."""
    print("\n" + "=" * 60)
    print("  PHASE 3: COLLECTION REPORT")
    print("=" * 60)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Count all images in uo_dataset
    uo_path = Path(UO_DATASET_DIR)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    camera_stats = {}
    hour_counts = defaultdict(int)
    total_images = 0

    for camera_dir in sorted(uo_path.iterdir()):
        if not camera_dir.is_dir():
            continue
        cam_name = camera_dir.name
        cam_images = 0
        cam_hours = defaultdict(int)

        for img in camera_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            cam_images += 1
            total_images += 1
            hour = parse_hour(img.name)
            if hour is not None:
                cam_hours[hour] += 1
                hour_counts[hour] += 1

        camera_stats[cam_name] = {
            "total_images": cam_images,
            "hours": dict(cam_hours),
        }

    # Save dataset overview CSV
    overview_path = Path(REPORT_DIR) / "dataset_overview.csv"
    with open(overview_path, "w") as f:
        f.write("camera,total_images,earliest_hour,latest_hour,hours_covered\n")
        for cam, stats in sorted(camera_stats.items(), key=lambda x: -x[1]["total_images"]):
            hours = stats["hours"]
            if hours:
                f.write(f"{cam},{stats['total_images']},"
                        f"{min(hours.keys())},{max(hours.keys())},"
                        f"{len(hours)}\n")

    # Save hourly distribution CSV
    hourly_path = Path(REPORT_DIR) / "hourly_distribution.csv"
    with open(hourly_path, "w") as f:
        f.write("hour,image_count,time_bucket\n")
        for h in range(24):
            bucket = get_time_bucket(h)
            f.write(f"{h:02d}:00,{hour_counts.get(h, 0)},{bucket}\n")

    # Save test set summary CSV
    test_path = Path(REPORT_DIR) / "test_set_summary.csv"
    with open(test_path, "w") as f:
        f.write("time_bucket,images,cameras,hours\n")
        for bucket in ["morning", "daytime", "evening", "night"]:
            imgs = [s for s in selected if s["bucket"] == bucket]
            cams = len(set(s["camera"] for s in imgs))
            hours = sorted(set(s["hour"] for s in imgs))
            hours_str = ";".join(str(h) for h in hours)
            f.write(f"{bucket},{len(imgs)},{cams},{hours_str}\n")

    # Print summary
    print(f"\n   Total dataset: {total_images} images across {len(camera_stats)} cameras")
    print(f"\n   Hourly distribution:")
    for h in range(24):
        count = hour_counts.get(h, 0)
        bar = "#" * (count // 50) if count > 0 else "-"
        bucket = get_time_bucket(h)
        print(f"     {h:02d}:00  {count:5d}  {bar:40s}  [{bucket}]")

    print(f"\n   Test set: {len(selected)} images from {len(selected_cameras)} cameras")
    for bucket in ["morning", "daytime", "evening", "night"]:
        count = sum(1 for s in selected if s["bucket"] == bucket)
        print(f"     {bucket:10s}: {count}")

    print(f"\n   Reports saved:")
    print(f"     {overview_path}")
    print(f"     {hourly_path}")
    print(f"     {test_path}")

    return total_images


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Final data collection for research")
    parser.add_argument("--end-hour", type=int, default=DEFAULT_END_HOUR,
                        help=f"Stop collecting at this hour (default: {DEFAULT_END_HOUR})")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip collection, just select test images from existing data")
    parser.add_argument("--interval", type=int, default=COLLECTION_INTERVAL,
                        help=f"Seconds between collection runs (default: {COLLECTION_INTERVAL})")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FINAL DATA COLLECTION FOR RESEARCH EVALUATION")
    print("  KF7029 MSc Project — Sumit Malviya (W24041293)")
    print("=" * 60)
    now = datetime.now()
    print(f"  Started:  {now.strftime('%A %d %B %Y, %H:%M')}")

    if not args.skip_collection:
        print(f"  End time: {args.end_hour}:00")
        hours_left = args.end_hour - now.hour
        runs_estimate = (hours_left * 3600) // args.interval
        print(f"  Estimated runs: ~{runs_estimate}")
        print(f"  Interval: {args.interval // 60} minutes")
    print(f"  Dataset:  {UO_DATASET_DIR}/")
    print("=" * 60)

    # Phase 1: Collection
    if not args.skip_collection:


        try:
            asyncio.run(collection_phase(args.end_hour))
        except KeyboardInterrupt:
            print("\n\n   Collection interrupted by user. Proceeding to selection...")
    else:
        print("\n   Skipping collection (--skip-collection flag)")

    # Phase 2: Select test images
    selected, selected_cameras = select_test_images()

    # Phase 3: Report
    total = generate_report(selected, selected_cameras)

    # Final instructions
    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print(f"""
  Your dataset: {total} images across {UO_DATASET_DIR}/
  Test set:     {len(selected)} images in {TEST_OUTPUT_DIR}/

  NEXT STEPS:
  1. Open Roboflow → Create new project (or add to existing)
  2. Upload {TEST_OUTPUT_DIR}/ folder (100 images)
  3. Annotate with 5 classes: Traffic light, Bus, Car, Person, Truck
  4. Export as YOLOv8 format → save to data/test_set_roboflow/
  5. Tell Claude "Fix 2 ready" → we build the proper evaluation notebook

  IMPORTANT:
  - These 100 images must NEVER go into training
  - The _manifest.csv tracks every image's source for reproducibility
  - The dataset reports in {REPORT_DIR}/ go in your paper's methodology

  TIME BUDGET:
  - Annotating 100 images: ~2-3 hours in Roboflow
  - Fix 2 (evaluation notebook): ~2 hours with Claude
  - Fix 3 (fine-tune eval): ~2 hours
  - Fix 4-8 (paper improvements): ~8 hours
  - Total to Monday: ~15 hours of work
""")
    print("=" * 60)


if __name__ == "__main__":
    main()