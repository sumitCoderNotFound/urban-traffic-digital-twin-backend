"""
Bulk collector V2: Download ALL Urban Observatory cameras every 5 minutes
Target: 20,000+ unique images
Saves into subfolders by camera location
Persistent hash tracking to survive restarts
"""
import asyncio
import aiohttp
import os
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path

CAMERA_URL = "https://portal.cctv.urbanobservatory.ac.uk/latest"
PHOTO_URL = "https://portal.cctv.urbanobservatory.ac.uk/photo"
BASE_DIR = Path("data/uo_dataset")
BASE_DIR.mkdir(parents=True, exist_ok=True)

HASH_FILE = BASE_DIR / "_seen_hashes.json"
LOG_FILE = BASE_DIR / "_collection_log.json"
TARGET = 20000
POLL_INTERVAL = 300  # 5 minutes

# Load existing hashes from disk
if HASH_FILE.exists():
    with open(HASH_FILE) as f:
        seen_hashes = set(json.load(f))
    print(f"Loaded {len(seen_hashes)} existing hashes from disk")
else:
    seen_hashes = set()

def save_hashes():
    with open(HASH_FILE, "w") as f:
        json.dump(list(seen_hashes), f)

def count_images():
    return len(list(BASE_DIR.rglob("*.jpg")))

def clean_folder_name(place_name):
    clean = re.sub(r"[^\w\s-]", "", place_name)
    clean = re.sub(r"\s+", "_", clean.strip())[:60]
    return clean if clean else "unknown"

def save_log(camera_stats):
    log = {
        "last_updated": datetime.now().isoformat(),
        "total_images": count_images(),
        "total_hashes": len(seen_hashes),
        "cameras": camera_stats
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

total_duplicates = 0
run_count = 0
camera_stats = {}

async def fetch_and_save():
    global total_duplicates, run_count, camera_stats
    run_count += 1

    timestamp = datetime.now()
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    current_count = count_images()

    print(f"\n{'='*60}")
    print(f"  RUN #{run_count} | {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Images on disk: {current_count} | Target: {TARGET}")
    print(f"{'='*60}")

    headers = {
        'User-Agent': 'UrbanDigitalTwin/1.0 (MSc Research - Northumbria University)',
    }

    timeout = aiohttp.ClientTimeout(total=60)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(CAMERA_URL, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    print(f"  ❌ API returned {resp.status}")
                    return
                cameras = await resp.json(content_type=None)

            if not isinstance(cameras, list):
                cameras = cameras.get('cameras', cameras.get('items', []))

            print(f"  📡 {len(cameras)} cameras from API")

            saved_this_run = 0
            skipped_this_run = 0
            errors = 0

            for idx, cam in enumerate(cameras):
                safe_path = cam.get('safe_photo_path', '')
                if not safe_path:
                    continue

                image_url = f"{PHOTO_URL}/{safe_path}"
                place = cam.get('place', '') or cam.get('short_description', 'unknown')

                # Create subfolder for this camera location
                folder_name = clean_folder_name(place)
                cam_dir = BASE_DIR / folder_name
                cam_dir.mkdir(parents=True, exist_ok=True)

                # Unique filename: timestamp + camera index + short hash of path
                path_hash = hashlib.md5(safe_path.encode()).hexdigest()[:6]
                filename = f"{time_str}_{path_hash}.jpg"
                filepath = cam_dir / filename

                try:
                    async with session.get(image_url, headers=headers, ssl=False) as img_resp:
                        if img_resp.status != 200:
                            errors += 1
                            continue
                        content = await img_resp.read()

                        if len(content) < 5000:
                            continue

                        img_hash = hashlib.md5(content).hexdigest()
                        if img_hash in seen_hashes:
                            skipped_this_run += 1
                            total_duplicates += 1
                            continue

                        seen_hashes.add(img_hash)

                        with open(filepath, "wb") as f:
                            f.write(content)

                        saved_this_run += 1

                        if folder_name not in camera_stats:
                            camera_stats[folder_name] = {"place": place, "images": 0}
                        camera_stats[folder_name]["images"] += 1

                except asyncio.TimeoutError:
                    errors += 1
                    continue
                except Exception:
                    errors += 1
                    continue

            save_hashes()
            save_log(camera_stats)

            new_total = count_images()
            print(f"  ✅ New this run: {saved_this_run}")
            print(f"  ⏭️  Duplicates skipped: {skipped_this_run}")
            if errors > 0:
                print(f"  ⚠️  Errors: {errors}")
            print(f"  📊 TOTAL ON DISK: {new_total} images")
            print(f"  📁 Camera folders: {len(camera_stats)}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

async def main():
    print("=" * 60)
    print("  BULK DATA COLLECTOR V2 - BY LOCATION")
    print("  Newcastle Urban Observatory")
    print(f"  Target: {TARGET} images")
    print(f"  Poll interval: {POLL_INTERVAL // 60} minutes")
    print(f"  Save directory: {BASE_DIR}")
    print(f"  Images already on disk: {count_images()}")
    print("=" * 60)

    while True:
        await fetch_and_save()

        current = count_images()
        if current >= TARGET:
            print(f"\n🎉 TARGET REACHED: {current} images on disk!")
            break

        remaining = TARGET - current
        print(f"\n  ⏳ Next run in {POLL_INTERVAL // 60} mins | {remaining} remaining")
        await asyncio.sleep(POLL_INTERVAL)

    final_count = count_images()
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"  Total images on disk: {final_count}")
    print(f"  Duplicates skipped: {total_duplicates}")
    print(f"  Camera locations: {len(camera_stats)}")
    print(f"  Saved to: {BASE_DIR}")
    print(f"\n  Top 10 cameras by image count:")
    sorted_cams = sorted(camera_stats.items(), key=lambda x: x[1]["images"], reverse=True)
    for name, info in sorted_cams[:10]:
        print(f"    {name:<50} {info['images']:>5} images")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())