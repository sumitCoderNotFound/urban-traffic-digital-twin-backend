"""
auto_label_v2.py - Auto-Label All 20k Newcastle Images
=======================================================
Project: Real-Time Traffic State Estimation Using Deep Learning
Author:  Sumit Malviya (W24041293)

WHAT THIS DOES:
- Runs YOLOv8n (COCO pre-trained) on ALL images in data/uo_dataset/
- Detects ALL 80 classes: cars, buses, trucks, pedestrians, cyclists, etc.
- Saves YOLO-format .txt label files (for fine-tuning later)
- Saves a research CSV (for analysis, charts, RQ1 evaluation)
- Saves per-camera summary stats

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/auto_label_v2.py

OPTIONS:
    python scripts/auto_label_v2.py --conf 0.25 --batch 64 --device mps
    python scripts/auto_label_v2.py --conf 0.30 --device cpu
    python scripts/auto_label_v2.py --sample 500    # test on 500 images first
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ── Parse arguments ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Auto-label Newcastle CCTV images")
parser.add_argument("--model",   default="yolov8n.pt",        help="Model weights path")
parser.add_argument("--images",  default="data/uo_dataset",   help="Input images directory")
parser.add_argument("--output",  default="data/labels",       help="Output labels directory")
parser.add_argument("--results", default="data/results",      help="Output CSV/JSON directory")
parser.add_argument("--conf",    type=float, default=0.25,    help="Confidence threshold (0.0-1.0)")
parser.add_argument("--batch",   type=int,   default=64,      help="Batch size")
parser.add_argument("--device",  default="mps",               help="Device: mps / cpu / 0 (NVIDIA)")
parser.add_argument("--imgsz",   type=int,   default=640,     help="Inference image size")
parser.add_argument("--sample",  type=int,   default=0,       help="Process only N images (0 = all)")
args = parser.parse_args()

# ── Import ultralytics after args (so --help works without it) ──
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ── COCO class names ─────────────────────────────────────────
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Traffic-relevant class IDs
VEHICLE_IDS     = {2, 3, 5, 7}       # car, motorcycle, bus, truck
PEDESTRIAN_IDS  = {0}                 # person
CYCLIST_IDS     = {1}                 # bicycle
TRAFFIC_IDS     = {9, 11}            # traffic light, stop sign

# ── Setup directories ────────────────────────────────────────
os.makedirs(args.output, exist_ok=True)
os.makedirs(args.results, exist_ok=True)

# ── Find all images ──────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_images = sorted([
    p for p in Path(args.images).rglob("*")
    if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
])

if args.sample > 0:
    all_images = all_images[:args.sample]
    print(f"⚠ SAMPLE MODE: Processing only {args.sample} images\n")

if not all_images:
    print(f"ERROR: No images found in {args.images}")
    sys.exit(1)

# ── Load model ───────────────────────────────────────────────
print("=" * 60)
print("  NEWCASTLE URBAN DIGITAL TWIN - AUTO LABELLER")
print("=" * 60)
print(f"  Model:       {args.model}")
print(f"  Images:      {len(all_images)}")
print(f"  Confidence:  {args.conf}")
print(f"  Batch size:  {args.batch}")
print(f"  Device:      {args.device}")
print(f"  Image size:  {args.imgsz}")
print(f"  Labels dir:  {args.output}")
print(f"  Results dir: {args.results}")
print("=" * 60)

model = YOLO(args.model)

# ── Process in batches ───────────────────────────────────────
all_rows = []
camera_stats = {}
total_batches = (len(all_images) + args.batch - 1) // args.batch
start_time = time.time()

for batch_start in range(0, len(all_images), args.batch):
    batch_files = all_images[batch_start:batch_start + args.batch]
    batch_num = batch_start // args.batch + 1
    
    # Run inference
    results = model.predict(
        source=[str(f) for f in batch_files],
        conf=args.conf,
        verbose=False,
        device=args.device,
        imgsz=args.imgsz,
    )
    
    batch_obj_count = 0
    
    for result, img_path in zip(results, batch_files):
        # Camera name = parent folder name
        camera = img_path.parent.name
        if camera == Path(args.images).name:
            camera = "root"
        
        # Create label subfolder per camera
        label_dir = Path(args.output) / camera
        os.makedirs(label_dir, exist_ok=True)
        label_file = label_dir / f"{img_path.stem}.txt"
        
        # Extract all detections
        label_lines = []
        counts = {}
        confs = []
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x, y, w, h = box.xywhn[0].tolist()
            
            # Save YOLO format: class_id x_center y_center width height
            label_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
            name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"cls_{cls_id}"
            counts[name] = counts.get(name, 0) + 1
            confs.append(conf)
        
        # Write label file
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))
        
        # Calculate group totals
        n_vehicles    = sum(1 for b in result.boxes if int(b.cls[0]) in VEHICLE_IDS)
        n_pedestrians = sum(1 for b in result.boxes if int(b.cls[0]) in PEDESTRIAN_IDS)
        n_cyclists    = sum(1 for b in result.boxes if int(b.cls[0]) in CYCLIST_IDS)
        n_traffic     = sum(1 for b in result.boxes if int(b.cls[0]) in TRAFFIC_IDS)
        n_total       = len(label_lines)
        avg_conf      = sum(confs) / len(confs) if confs else 0.0
        
        batch_obj_count += n_total
        
        # Build CSV row
        row = {
            "image":           img_path.name,
            "camera":          camera,
            "total_objects":   n_total,
            "vehicles":        n_vehicles,
            "cars":            counts.get("car", 0),
            "buses":           counts.get("bus", 0),
            "trucks":          counts.get("truck", 0),
            "motorcycles":     counts.get("motorcycle", 0),
            "pedestrians":     n_pedestrians,
            "cyclists":        n_cyclists,
            "traffic_lights":  counts.get("traffic light", 0),
            "stop_signs":      counts.get("stop sign", 0),
            "avg_confidence":  round(avg_conf, 4),
            "all_detections":  json.dumps(counts) if counts else "{}",
        }
        all_rows.append(row)
        
        # Update camera stats
        if camera not in camera_stats:
            camera_stats[camera] = {
                "images": 0, "vehicles": 0, "pedestrians": 0,
                "cyclists": 0, "total_objects": 0
            }
        camera_stats[camera]["images"] += 1
        camera_stats[camera]["vehicles"] += n_vehicles
        camera_stats[camera]["pedestrians"] += n_pedestrians
        camera_stats[camera]["cyclists"] += n_cyclists
        camera_stats[camera]["total_objects"] += n_total
    
    # Progress
    elapsed = time.time() - start_time
    imgs_done = batch_start + len(batch_files)
    imgs_per_sec = imgs_done / elapsed if elapsed > 0 else 0
    remaining = (len(all_images) - imgs_done) / imgs_per_sec if imgs_per_sec > 0 else 0
    
    print(
        f"  [{batch_num:3d}/{total_batches}] "
        f"{imgs_done:6d}/{len(all_images)} images | "
        f"{batch_obj_count:4d} objects | "
        f"{imgs_per_sec:.1f} img/s | "
        f"~{remaining/60:.0f} min left"
    )

# ── Save CSV ─────────────────────────────────────────────────
csv_path = Path(args.results) / "auto_label_detections.csv"
if all_rows:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

# ── Save per-camera summary ──────────────────────────────────
camera_csv_path = Path(args.results) / "auto_label_camera_summary.csv"
cam_rows = []
for cam, stats in sorted(camera_stats.items(), key=lambda x: x[1]["images"], reverse=True):
    n = stats["images"]
    cam_rows.append({
        "camera": cam,
        "images": n,
        "total_objects": stats["total_objects"],
        "vehicles": stats["vehicles"],
        "pedestrians": stats["pedestrians"],
        "cyclists": stats["cyclists"],
        "avg_vehicles_per_image": round(stats["vehicles"] / n, 2) if n else 0,
        "avg_pedestrians_per_image": round(stats["pedestrians"] / n, 2) if n else 0,
        "avg_cyclists_per_image": round(stats["cyclists"] / n, 2) if n else 0,
    })

with open(camera_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cam_rows[0].keys())
    writer.writeheader()
    writer.writerows(cam_rows)

# ── Save full JSON ───────────────────────────────────────────
json_path = Path(args.results) / "auto_label_full.json"
with open(json_path, "w") as f:
    json.dump({
        "config": {
            "model": args.model,
            "confidence": args.conf,
            "image_size": args.imgsz,
            "total_images": len(all_images),
        },
        "camera_summary": camera_stats,
        "detections": all_rows,
    }, f, indent=2)

# ── Print final summary ──────────────────────────────────────
total_time = time.time() - start_time
total_obj  = sum(r["total_objects"] for r in all_rows)
total_veh  = sum(r["vehicles"] for r in all_rows)
total_ped  = sum(r["pedestrians"] for r in all_rows)
total_cyc  = sum(r["cyclists"] for r in all_rows)
total_cars = sum(r["cars"] for r in all_rows)
total_bus  = sum(r["buses"] for r in all_rows)
total_trk  = sum(r["trucks"] for r in all_rows)
total_moto = sum(r["motorcycles"] for r in all_rows)
total_tl   = sum(r["traffic_lights"] for r in all_rows)
imgs_w_veh = sum(1 for r in all_rows if r["vehicles"] > 0)
imgs_w_ped = sum(1 for r in all_rows if r["pedestrians"] > 0)
imgs_w_cyc = sum(1 for r in all_rows if r["cyclists"] > 0)
avg_conf   = sum(r["avg_confidence"] for r in all_rows) / len(all_rows) if all_rows else 0

print(f"\n{'=' * 60}")
print(f"  AUTO-LABEL COMPLETE")
print(f"{'=' * 60}")
print(f"  Time taken:            {total_time/60:.1f} minutes")
print(f"  Images processed:      {len(all_rows)}")
print(f"  Cameras:               {len(camera_stats)}")
print(f"{'─' * 60}")
print(f"  TOTAL OBJECTS:         {total_obj}")
print(f"{'─' * 60}")
print(f"  Vehicles:              {total_veh}")
print(f"    ├ Cars:              {total_cars}")
print(f"    ├ Buses:             {total_bus}")
print(f"    ├ Trucks:            {total_trk}")
print(f"    └ Motorcycles:       {total_moto}")
print(f"  Pedestrians:           {total_ped}")
print(f"  Cyclists:              {total_cyc}")
print(f"  Traffic lights:        {total_tl}")
print(f"{'─' * 60}")
print(f"  Images with vehicles:  {imgs_w_veh}/{len(all_rows)} ({100*imgs_w_veh/len(all_rows):.1f}%)")
print(f"  Images with people:    {imgs_w_ped}/{len(all_rows)} ({100*imgs_w_ped/len(all_rows):.1f}%)")
print(f"  Images with cyclists:  {imgs_w_cyc}/{len(all_rows)} ({100*imgs_w_cyc/len(all_rows):.1f}%)")
print(f"  Avg confidence:        {avg_conf:.3f}")
print(f"{'─' * 60}")
print(f"  TOP 10 CAMERAS BY ACTIVITY:")
for cam in cam_rows[:10]:
    print(f"    {cam['camera']:40s}  {cam['avg_vehicles_per_image']:5.1f} veh/img  {cam['avg_pedestrians_per_image']:5.1f} ped/img")
print(f"{'=' * 60}")
print(f"  Labels:         {args.output}/")
print(f"  Detections CSV: {csv_path}")
print(f"  Camera CSV:     {camera_csv_path}")
print(f"  Full JSON:      {json_path}")
print(f"{'=' * 60}")
print(f"\n  NEXT STEPS:")
print(f"  1. Open the CSV in Excel/Python to explore detection patterns")
print(f"  2. Select 250 images for manual ground truth annotation")
print(f"  3. Run RQ1_Baseline_Evaluation.ipynb to measure accuracy")
print(f"{'=' * 60}")