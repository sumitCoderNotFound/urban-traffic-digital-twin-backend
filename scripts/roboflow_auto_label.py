"""
Roboflow Auto-Label via API
============================
1. Downloads unannotated images from Roboflow
2. Runs local YOLOv8s fine-tuned model on each image
3. Uploads bounding box annotations back to Roboflow

Usage:
    python scripts/roboflow_auto_label.py --api-key YOUR_KEY

Author: Sumit Malviya (W24041293)
"""

import os
import re
import sys
import json
import time
import argparse
import requests
import tempfile
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
WORKSPACE   = "sumits-workspace-dnomt"
PROJECT     = "urbandigititaltwin"
MODEL_PATH  = "runs/detect/newcastle_v6_improved/weights/best.pt"
CONF        = 0.25
IMGSZ       = 960
DEVICE      = "cpu"
BATCH_SIZE  = 50   # process 50 images at a time

# Class names must match Roboflow project exactly
RF_NAMES = {
    0: "Motorcycle",
    1: "Roadwork",
    2: "Traffic light",
    3: "bicycle",
    4: "bus",
    5: "car",
    6: "person",
    7: "truck"
}

# ============================================================
# ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description="Auto-label Roboflow images")
parser.add_argument("--api-key", required=True, help="Roboflow private API key")
parser.add_argument("--limit",   type=int, default=0, help="Max images to process (0=all)")
parser.add_argument("--conf",    type=float, default=CONF)
parser.add_argument("--skip-cars", action="store_true",
                    help="Skip uploading car/person/traffic light labels (already have enough)")
args = parser.parse_args()

API_KEY = args.api_key
BASE_URL = f"https://api.roboflow.com"

# ============================================================
# ROBOFLOW API HELPERS
# ============================================================
def get_project_info():
    url = f"{BASE_URL}/{WORKSPACE}/{PROJECT}?api_key={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"ERROR: Could not access project: {r.text}")
        sys.exit(1)
    return r.json()

def get_unannotated_images(limit=100, offset=0):
    url = (f"{BASE_URL}/{WORKSPACE}/{PROJECT}/images"
           f"?api_key={API_KEY}&split=train&annotated=false"
           f"&limit={limit}&offset={offset}")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"ERROR getting images: {r.text}")
        return []
    data = r.json()
    return data.get("images", [])

def upload_annotation(image_id, image_w, image_h, boxes):
    """Upload YOLO format annotations to Roboflow as bounding boxes."""
    if not boxes:
        return True

    annotations = []
    for box in boxes:
        cls_id = box["class"]
        cls_name = RF_NAMES.get(cls_id, f"class_{cls_id}")

        # Skip majority classes if flag set
        if args.skip_cars and cls_id in {2, 5, 6}:  # traffic light, car, person
            continue

        # Convert YOLO normalized xywh to pixel coordinates
        cx = box["box"][0] * image_w
        cy = box["box"][1] * image_h
        w  = box["box"][2] * image_w
        h  = box["box"][3] * image_h
        x1 = cx - w / 2
        y1 = cy - h / 2

        annotations.append({
            "label": cls_name,
            "x": round(x1, 2),
            "y": round(y1, 2),
            "width":  round(w, 2),
            "height": round(h, 2),
        })

    if not annotations:
        return True

    url = (f"{BASE_URL}/dataset/{WORKSPACE}/{PROJECT}"
           f"/annotate/{image_id}?api_key={API_KEY}&name=autolabel")

    payload = {"annotations": json.dumps(annotations)}
    r = requests.post(url, data=payload)

    if r.status_code not in (200, 201):
        print(f"  WARNING: Upload failed for {image_id}: {r.text[:100]}")
        return False
    return True

# ============================================================
# MAIN
# ============================================================
print(f"\n{'='*60}")
print(f"  ROBOFLOW AUTO-LABELLER")
print(f"  Workspace: {WORKSPACE}")
print(f"  Project:   {PROJECT}")
print(f"  Model:     {MODEL_PATH}")
print(f"  Conf:      {args.conf}")
print(f"{'='*60}\n")

# Verify project access
print("  Checking project access...")
info = get_project_info()
print(f"  Project: {info.get('project', {}).get('name', 'unknown')}")

# Load model
print(f"\n  Loading model: {MODEL_PATH}")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
model = YOLO(MODEL_PATH)
print(f"  Model loaded — {len(RF_NAMES)} classes")

# Get all unannotated images
print(f"\n  Fetching unannotated images...")
all_images = []
offset = 0
while True:
    batch = get_unannotated_images(limit=100, offset=offset)
    if not batch:
        break
    all_images.extend(batch)
    print(f"    Fetched {len(all_images)} images...")
    if len(batch) < 100:
        break
    offset += 100
    time.sleep(0.5)

if args.limit > 0:
    all_images = all_images[:args.limit]

print(f"  Total unannotated images: {len(all_images)}")

if not all_images:
    print("  No unannotated images found. All done!")
    sys.exit(0)

# Process images
processed = 0
uploaded  = 0
skipped   = 0
errors    = 0

print(f"\n  Processing {len(all_images)} images...\n")

with tempfile.TemporaryDirectory() as tmpdir:
    for i, img_info in enumerate(all_images):
        img_id  = img_info.get("id", "")
        img_url = img_info.get("url", "") or img_info.get("image", {}).get("original", "")
        img_name = img_info.get("name", f"image_{i}")
        img_w   = img_info.get("width",  640)
        img_h   = img_info.get("height", 480)

        if not img_url:
            skipped += 1
            continue

        # Download image
        tmp_path = os.path.join(tmpdir, f"img_{i}.jpg")
        try:
            urlretrieve(img_url, tmp_path)
        except Exception as e:
            print(f"  [{i+1}/{len(all_images)}] Download failed: {img_name}: {e}")
            errors += 1
            continue

        # Run detection
        try:
            results = model.predict(
                source=tmp_path,
                conf=args.conf,
                verbose=False,
                device=DEVICE,
                imgsz=IMGSZ
            )
        except Exception as e:
            print(f"  [{i+1}/{len(all_images)}] Detection failed: {e}")
            errors += 1
            continue

        # Parse detections
        boxes = []
        for b in results[0].boxes:
            cid = int(b.cls[0])
            if cid in RF_NAMES:
                boxes.append({
                    "class": cid,
                    "box":   b.xywhn[0].tolist(),
                    "conf":  float(b.conf[0])
                })

        # Upload annotations
        success = upload_annotation(img_id, img_w, img_h, boxes)
        if success:
            uploaded += 1
        else:
            errors += 1

        processed += 1

        # Progress
        if (i + 1) % 25 == 0 or (i + 1) == len(all_images):
            print(f"  [{i+1:4d}/{len(all_images)}] "
                  f"uploaded={uploaded} skipped={skipped} errors={errors} "
                  f"detections={len(boxes)}")

        # Small delay to avoid rate limiting
        time.sleep(0.1)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
print(f"  Processed : {processed}")
print(f"  Uploaded  : {uploaded}")
print(f"  Skipped   : {skipped}")
print(f"  Errors    : {errors}")
print(f"\n  Go to Roboflow → Annotate to review the labels")
print(f"  Then generate a new version and export for retraining")
print(f"{'='*60}")
