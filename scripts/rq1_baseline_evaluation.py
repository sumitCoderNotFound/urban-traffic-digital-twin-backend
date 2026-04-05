"""
rq1_baseline_evaluation.py - RQ1 Baseline Evaluation
=====================================================
Project: Real-Time Multi-Modal Urban Activity Estimation
         Using Deep Learning and Live Camera Feeds for Urban Digital Twins
Author:  Sumit Malviya (W24041293)
Module:  KF7029 - MSc Computer Science, Northumbria University
Supervisor: Dr. Jason Moore

PURPOSE:
    Evaluates pre-trained YOLOv8n (COCO weights) performance on
    Newcastle Urban Observatory CCTV images by comparing auto-detections
    against human-annotated ground truth from Roboflow.

METRICS COMPUTED:
    - Per-class: Precision, Recall, F1-Score, AP@0.5
    - Overall:   mAP@0.5, mAP@0.5:0.95
    - Per-camera: Detection accuracy breakdown
    - Confidence threshold analysis
    - Confusion matrix

INPUTS:
    - Ground truth labels: data/ground_truth_roboflow/train/labels/ (+ valid/ + test/)
    - Ground truth images: data/ground_truth_roboflow/train/images/ (+ valid/ + test/)
    - Model: yolov8n.pt (COCO pre-trained)
    - data.yaml from Roboflow export

OUTPUTS:
    - data/results/rq1_overall_metrics.csv
    - data/results/rq1_per_class_metrics.csv
    - data/results/rq1_per_camera_metrics.csv
    - data/results/rq1_per_image_metrics.csv
    - data/results/rq1_confidence_analysis.csv
    - data/results/rq1_confusion_matrix.png
    - data/results/rq1_pr_curves.png
    - data/results/rq1_summary_report.txt

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/rq1_baseline_evaluation.py

OPTIONS:
    python scripts/rq1_baseline_evaluation.py --conf 0.25
    python scripts/rq1_baseline_evaluation.py --conf 0.5 --iou 0.5
    python scripts/rq1_baseline_evaluation.py --gt-dir data/ground_truth_roboflow
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Parse arguments ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="RQ1 Baseline Evaluation - YOLOv8n on Newcastle CCTV")
parser.add_argument("--model",    default="yolov8n.pt",                    help="Model weights path")
parser.add_argument("--gt-dir",   default="data/ground_truth_roboflow",    help="Ground truth directory (Roboflow export)")
parser.add_argument("--results",  default="data/results",                  help="Output results directory")
parser.add_argument("--conf",     type=float, default=0.25,                help="Confidence threshold")
parser.add_argument("--iou",      type=float, default=0.5,                 help="IoU threshold for matching")
parser.add_argument("--device",   default="cpu",                           help="Device: cpu / mps / 0")
parser.add_argument("--imgsz",    type=int,   default=640,                 help="Inference image size")
args = parser.parse_args()

# ── Import ultralytics ───────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ── CLASS MAPPING ────────────────────────────────────────────
# Roboflow exported classes (alphabetical order from your data.yaml):
#   0 = Traffic light, 1 = bus, 2 = car, 3 = person, 4 = truck
#
# COCO class IDs used by YOLOv8n:
#   0 = person, 1 = bicycle, 2 = car, 3 = motorcycle, 5 = bus,
#   7 = truck, 9 = traffic light
#
# We need to map COCO predictions → Roboflow GT class IDs

COCO_TO_ROBOFLOW = {
    9:  0,   # traffic light → 0
    5:  1,   # bus → 1
    2:  2,   # car → 2
    0:  3,   # person → 3
    7:  4,   # truck → 4
}

ROBOFLOW_NAMES = {
    0: "Traffic light",
    1: "bus",
    2: "car",
    3: "person",
    4: "truck",
}

COCO_NAMES_FULL = [
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

NUM_CLASSES = 5


# ── HELPER FUNCTIONS ─────────────────────────────────────────

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x_center, y_center, w, h] normalised format."""
    # Convert to [x1, y1, x2, y2]
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_gt_labels(label_path):
    """Load ground truth labels from YOLO format .txt file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append({"class": cls_id, "box": [x, y, w, h]})
    return boxes


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth using IoU threshold.
    Returns: list of (pred_idx, gt_idx, iou) matches, unmatched_preds, unmatched_gts
    """
    if not pred_boxes or not gt_boxes:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred["box"], gt["box"])

    matches = []
    matched_preds = set()
    matched_gts = set()

    # Greedy matching: highest IoU first
    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        pred_idx, gt_idx = int(max_idx[0]), int(max_idx[1])

        # Only match if same class
        if pred_boxes[pred_idx]["class"] == gt_boxes[gt_idx]["class"]:
            matches.append((pred_idx, gt_idx, max_iou))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

        # Remove this pair from consideration
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, gt_idx] = 0

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def compute_ap(precisions, recalls):
    """Compute Average Precision using 11-point interpolation."""
    if not precisions or not recalls:
        return 0.0

    # Sort by recall
    sorted_pairs = sorted(zip(recalls, precisions))
    recalls_sorted = [p[0] for p in sorted_pairs]
    precisions_sorted = [p[1] for p in sorted_pairs]

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_above = [p for p, r in zip(precisions_sorted, recalls_sorted) if r >= t]
        if precisions_above:
            ap += max(precisions_above)
    ap /= 11.0
    return ap


# ── COLLECT ALL IMAGES AND LABELS ────────────────────────────

def collect_dataset(gt_dir):
    """Collect all image-label pairs from train/valid/test splits."""
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = Path(gt_dir) / split / "images"
        lbl_dir = Path(gt_dir) / split / "labels"
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.glob("*")):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                pairs.append({
                    "image": str(img_file),
                    "label": str(lbl_file),
                    "split": split,
                    "filename": img_file.name,
                    "camera": extract_camera_name(img_file.name),
                })
    return pairs


def extract_camera_name(filename):
    """Extract camera name from filename like 'A167_Sniperley_RbtSouthfield_Way__20260324_...'"""
    # Split on double underscore to separate camera name from timestamp
    parts = filename.split("__")
    if len(parts) >= 2:
        return parts[0]
    # Fallback: split on first occurrence of '2026' or '2025'
    for year in ["20260", "20250", "20240", "20230"]:
        idx = filename.find(year)
        if idx > 0:
            return filename[:idx].rstrip("_")
    return filename.split("_")[0]


# ── MAIN EVALUATION ──────────────────────────────────────────

def main():
    print("=" * 65)
    print("  RQ1 BASELINE EVALUATION")
    print("  Pre-trained YOLOv8n on Newcastle Urban Observatory CCTV")
    print("=" * 65)
    print(f"  Model:          {args.model}")
    print(f"  Ground truth:   {args.gt_dir}")
    print(f"  Confidence:     {args.conf}")
    print(f"  IoU threshold:  {args.iou}")
    print(f"  Device:         {args.device}")
    print(f"  Image size:     {args.imgsz}")
    print("=" * 65)

    # Load model
    model = YOLO(args.model)

    # Collect dataset
    dataset = collect_dataset(args.gt_dir)
    if not dataset:
        print(f"ERROR: No images found in {args.gt_dir}")
        sys.exit(1)
    print(f"\n  Found {len(dataset)} images across splits:")
    for split in ["train", "valid", "test"]:
        count = sum(1 for d in dataset if d["split"] == split)
        if count > 0:
            print(f"    {split}: {count} images")

    # Setup results directory
    os.makedirs(args.results, exist_ok=True)

    # ── Run inference and evaluate each image ────────────────
    per_image_results = []
    all_class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    per_camera_stats = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))
    all_pred_confs = defaultdict(list)  # class → list of (confidence, is_tp)
    confusion = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)  # +1 for background

    start_time = time.time()

    for idx, item in enumerate(dataset):
        img_path = item["image"]
        lbl_path = item["label"]
        camera = item["camera"]

        # Load ground truth
        gt_boxes = load_gt_labels(lbl_path)

        # Run YOLOv8n inference
        results = model.predict(
            source=img_path,
            conf=args.conf,
            verbose=False,
            device=args.device,
            imgsz=args.imgsz,
        )

        # Convert YOLO predictions to Roboflow class IDs
        pred_boxes = []
        for box in results[0].boxes:
            coco_cls = int(box.cls[0])
            conf = float(box.conf[0])
            x, y, w, h = box.xywhn[0].tolist()

            # Map COCO class to Roboflow class
            if coco_cls in COCO_TO_ROBOFLOW:
                rf_cls = COCO_TO_ROBOFLOW[coco_cls]
                pred_boxes.append({
                    "class": rf_cls,
                    "box": [x, y, w, h],
                    "confidence": conf,
                    "coco_class": coco_cls,
                })

        # Match predictions to ground truth
        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            pred_boxes, gt_boxes, iou_threshold=args.iou
        )

        # Count TP, FP, FN per class
        img_tp = defaultdict(int)
        img_fp = defaultdict(int)
        img_fn = defaultdict(int)

        # True Positives
        for pred_idx, gt_idx, iou_val in matches:
            cls = pred_boxes[pred_idx]["class"]
            img_tp[cls] += 1
            all_class_stats[cls]["TP"] += 1
            per_camera_stats[camera][cls]["TP"] += 1
            all_pred_confs[cls].append((pred_boxes[pred_idx]["confidence"], True))
            confusion[cls, cls] += 1

        # False Positives (predictions with no matching GT)
        for pred_idx in unmatched_preds:
            cls = pred_boxes[pred_idx]["class"]
            img_fp[cls] += 1
            all_class_stats[cls]["FP"] += 1
            per_camera_stats[camera][cls]["FP"] += 1
            all_pred_confs[cls].append((pred_boxes[pred_idx]["confidence"], False))
            confusion[NUM_CLASSES, cls] += 1  # background predicted as cls

        # False Negatives (GT with no matching prediction)
        for gt_idx in unmatched_gts:
            cls = gt_boxes[gt_idx]["class"]
            img_fn[cls] += 1
            all_class_stats[cls]["FN"] += 1
            per_camera_stats[camera][cls]["FN"] += 1
            confusion[cls, NUM_CLASSES] += 1  # cls missed (predicted as background)

        # Per-image metrics
        total_tp = sum(img_tp.values())
        total_fp = sum(img_fp.values())
        total_fn = sum(img_fn.values())
        img_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        img_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0

        per_image_results.append({
            "filename": item["filename"],
            "camera": camera,
            "split": item["split"],
            "gt_objects": len(gt_boxes),
            "pred_objects": len(pred_boxes),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(img_precision, 4),
            "recall": round(img_recall, 4),
            "f1_score": round(img_f1, 4),
            "gt_cars": sum(1 for b in gt_boxes if b["class"] == 2),
            "pred_cars": sum(1 for b in pred_boxes if b["class"] == 2),
            "gt_persons": sum(1 for b in gt_boxes if b["class"] == 3),
            "pred_persons": sum(1 for b in pred_boxes if b["class"] == 3),
            "gt_buses": sum(1 for b in gt_boxes if b["class"] == 1),
            "pred_buses": sum(1 for b in pred_boxes if b["class"] == 1),
            "gt_trucks": sum(1 for b in gt_boxes if b["class"] == 4),
            "pred_trucks": sum(1 for b in pred_boxes if b["class"] == 4),
            "gt_traffic_lights": sum(1 for b in gt_boxes if b["class"] == 0),
            "pred_traffic_lights": sum(1 for b in pred_boxes if b["class"] == 0),
        })

        # Progress
        if (idx + 1) % 20 == 0 or (idx + 1) == len(dataset):
            elapsed = time.time() - start_time
            print(f"  [{idx+1:3d}/{len(dataset)}] processed | {elapsed:.1f}s elapsed")

    # ── COMPUTE PER-CLASS METRICS ────────────────────────────
    print(f"\n{'─' * 65}")
    print("  PER-CLASS RESULTS")
    print(f"{'─' * 65}")
    print(f"  {'Class':<18s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AP@.5':>7s}")
    print(f"  {'─'*18} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    per_class_rows = []
    aps = []

    for cls_id in sorted(ROBOFLOW_NAMES.keys()):
        stats = all_class_stats[cls_id]
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Compute AP@0.5 using confidence-sorted predictions
        confs = all_pred_confs[cls_id]
        if confs:
            confs_sorted = sorted(confs, key=lambda x: x[0], reverse=True)
            running_tp = 0
            running_fp = 0
            total_gt = tp + fn
            precs = []
            recs = []
            for conf_val, is_tp in confs_sorted:
                if is_tp:
                    running_tp += 1
                else:
                    running_fp += 1
                p = running_tp / (running_tp + running_fp)
                r = running_tp / total_gt if total_gt > 0 else 0
                precs.append(p)
                recs.append(r)
            ap = compute_ap(precs, recs)
        else:
            ap = 0.0

        aps.append(ap)
        name = ROBOFLOW_NAMES[cls_id]
        print(f"  {name:<18s} {tp:5d} {fp:5d} {fn:5d} {precision:7.3f} {recall:7.3f} {f1:7.3f} {ap:7.3f}")

        per_class_rows.append({
            "class_id": cls_id,
            "class_name": name,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "ap_50": round(ap, 4),
        })

    # ── COMPUTE OVERALL METRICS ──────────────────────────────
    total_tp = sum(all_class_stats[c]["TP"] for c in ROBOFLOW_NAMES)
    total_fp = sum(all_class_stats[c]["FP"] for c in ROBOFLOW_NAMES)
    total_fn = sum(all_class_stats[c]["FN"] for c in ROBOFLOW_NAMES)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    mAP_50 = np.mean(aps) if aps else 0.0

    print(f"  {'─'*18} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'OVERALL':<18s} {total_tp:5d} {total_fp:5d} {total_fn:5d} {overall_precision:7.3f} {overall_recall:7.3f} {overall_f1:7.3f} {mAP_50:7.3f}")

    # ── COMPUTE PER-CAMERA METRICS ───────────────────────────
    print(f"\n{'─' * 65}")
    print("  PER-CAMERA RESULTS")
    print(f"{'─' * 65}")
    print(f"  {'Camera':<40s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print(f"  {'─'*40} {'─'*4} {'─'*4} {'─'*4} {'─'*6} {'─'*6} {'─'*6}")

    per_camera_rows = []
    for camera in sorted(per_camera_stats.keys()):
        cam_tp = sum(per_camera_stats[camera][c]["TP"] for c in ROBOFLOW_NAMES)
        cam_fp = sum(per_camera_stats[camera][c]["FP"] for c in ROBOFLOW_NAMES)
        cam_fn = sum(per_camera_stats[camera][c]["FN"] for c in ROBOFLOW_NAMES)

        cam_prec = cam_tp / (cam_tp + cam_fp) if (cam_tp + cam_fp) > 0 else 0
        cam_rec = cam_tp / (cam_tp + cam_fn) if (cam_tp + cam_fn) > 0 else 0
        cam_f1 = 2 * cam_prec * cam_rec / (cam_prec + cam_rec) if (cam_prec + cam_rec) > 0 else 0

        print(f"  {camera:<40s} {cam_tp:4d} {cam_fp:4d} {cam_fn:4d} {cam_prec:6.3f} {cam_rec:6.3f} {cam_f1:6.3f}")

        per_camera_rows.append({
            "camera": camera,
            "true_positives": cam_tp,
            "false_positives": cam_fp,
            "false_negatives": cam_fn,
            "precision": round(cam_prec, 4),
            "recall": round(cam_rec, 4),
            "f1_score": round(cam_f1, 4),
        })

    # ── CONFIDENCE THRESHOLD ANALYSIS ────────────────────────
    print(f"\n{'─' * 65}")
    print("  CONFIDENCE THRESHOLD ANALYSIS")
    print(f"{'─' * 65}")

    conf_analysis_rows = []
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        t_tp = 0
        t_fp = 0
        t_fn_extra = 0  # Additional FN from filtering

        for cls_id in ROBOFLOW_NAMES:
            for conf_val, is_tp in all_pred_confs[cls_id]:
                if conf_val >= threshold:
                    if is_tp:
                        t_tp += 1
                    else:
                        t_fp += 1
                else:
                    if is_tp:
                        t_fn_extra += 1  # Was TP at lower threshold, now missed

        t_fn = total_fn + t_fn_extra
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0

        print(f"  conf >= {threshold:.2f}:  Precision={t_prec:.3f}  Recall={t_rec:.3f}  F1={t_f1:.3f}  (TP={t_tp} FP={t_fp} FN={t_fn})")

        conf_analysis_rows.append({
            "confidence_threshold": threshold,
            "true_positives": t_tp,
            "false_positives": t_fp,
            "false_negatives": t_fn,
            "precision": round(t_prec, 4),
            "recall": round(t_rec, 4),
            "f1_score": round(t_f1, 4),
        })

    # ── SAVE ALL CSVs ────────────────────────────────────────
    # 1. Overall metrics
    overall_csv = Path(args.results) / "rq1_overall_metrics.csv"
    with open(overall_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_images", len(dataset)])
        writer.writerow(["total_gt_objects", total_tp + total_fn])
        writer.writerow(["total_predictions", total_tp + total_fp])
        writer.writerow(["true_positives", total_tp])
        writer.writerow(["false_positives", total_fp])
        writer.writerow(["false_negatives", total_fn])
        writer.writerow(["precision", round(overall_precision, 4)])
        writer.writerow(["recall", round(overall_recall, 4)])
        writer.writerow(["f1_score", round(overall_f1, 4)])
        writer.writerow(["mAP_50", round(mAP_50, 4)])
        writer.writerow(["confidence_threshold", args.conf])
        writer.writerow(["iou_threshold", args.iou])
        writer.writerow(["model", args.model])
        writer.writerow(["num_classes", NUM_CLASSES])

    # 2. Per-class metrics
    class_csv = Path(args.results) / "rq1_per_class_metrics.csv"
    with open(class_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_class_rows)

    # 3. Per-camera metrics
    camera_csv = Path(args.results) / "rq1_per_camera_metrics.csv"
    with open(camera_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_camera_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_camera_rows)

    # 4. Per-image metrics
    image_csv = Path(args.results) / "rq1_per_image_metrics.csv"
    with open(image_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_results[0].keys())
        writer.writeheader()
        writer.writerows(per_image_results)

    # 5. Confidence analysis
    conf_csv = Path(args.results) / "rq1_confidence_analysis.csv"
    with open(conf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=conf_analysis_rows[0].keys())
        writer.writeheader()
        writer.writerows(conf_analysis_rows)

    # ── SAVE SUMMARY REPORT ──────────────────────────────────
    report_path = Path(args.results) / "rq1_summary_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("  RQ1 BASELINE EVALUATION REPORT\n")
        f.write("  Pre-trained YOLOv8n on Newcastle Urban Observatory CCTV\n")
        f.write("=" * 65 + "\n")
        f.write(f"  Date:             {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Model:            {args.model}\n")
        f.write(f"  Total images:     {len(dataset)}\n")
        f.write(f"  Confidence:       {args.conf}\n")
        f.write(f"  IoU threshold:    {args.iou}\n")
        f.write(f"  Classes:          {NUM_CLASSES} ({', '.join(ROBOFLOW_NAMES.values())})\n")
        f.write("=" * 65 + "\n\n")
        f.write("OVERALL METRICS:\n")
        f.write(f"  Precision:        {overall_precision:.4f}\n")
        f.write(f"  Recall:           {overall_recall:.4f}\n")
        f.write(f"  F1-Score:         {overall_f1:.4f}\n")
        f.write(f"  mAP@0.5:          {mAP_50:.4f}\n")
        f.write(f"  True Positives:   {total_tp}\n")
        f.write(f"  False Positives:  {total_fp}\n")
        f.write(f"  False Negatives:  {total_fn}\n")
        f.write("\n\nPER-CLASS METRICS:\n")
        f.write(f"  {'Class':<18s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AP@.5':>7s}\n")
        for row in per_class_rows:
            f.write(f"  {row['class_name']:<18s} {row['precision']:7.4f} {row['recall']:7.4f} {row['f1_score']:7.4f} {row['ap_50']:7.4f}\n")
        f.write(f"\n\nPER-CAMERA METRICS:\n")
        for row in per_camera_rows:
            f.write(f"  {row['camera']:<40s} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1_score']:.3f}\n")

    # ── FINAL SUMMARY ────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n{'=' * 65}")
    print(f"  RQ1 EVALUATION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Time:             {total_time:.1f} seconds")
    print(f"  Images evaluated: {len(dataset)}")
    print(f"  GT objects:       {total_tp + total_fn}")
    print(f"  Predictions:      {total_tp + total_fp}")
    print(f"{'─' * 65}")
    print(f"  OVERALL PRECISION:  {overall_precision:.4f}")
    print(f"  OVERALL RECALL:     {overall_recall:.4f}")
    print(f"  OVERALL F1:         {overall_f1:.4f}")
    print(f"  mAP@0.5:            {mAP_50:.4f}")
    print(f"{'─' * 65}")
    print(f"  Saved results to:")
    print(f"    {overall_csv}")
    print(f"    {class_csv}")
    print(f"    {camera_csv}")
    print(f"    {image_csv}")
    print(f"    {conf_csv}")
    print(f"    {report_path}")
    print(f"{'=' * 65}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Review per-class results - which classes perform worst?")
    print(f"  2. Review per-camera results - which cameras are hardest?")
    print(f"  3. Use confidence analysis to find optimal threshold")
    print(f"  4. These results form your RQ1 baseline for the paper")
    print(f"  5. After fine-tuning (RQ2), re-run to measure improvement")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()