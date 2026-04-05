"""
RQ1_RQ2_FINAL_EVALUATION.py
============================
Proper evaluation on HELD-OUT TEST SET ONLY (78 images)
No training data contamination.

Evaluates:
  - RQ1: Baseline YOLOv8n (COCO pre-trained)
  - RQ2: Fine-tuned YOLOv8n (Newcastle-specific)

Metrics:
  - mAP@0.5 (standard)
  - mAP@0.5:0.95 (COCO standard)
  - Per-class P, R, F1, AP
  - Per-camera breakdown
  - Day vs Night performance
  - Confidence threshold analysis

HOW TO RUN:
    cd urban-digital-twin-backend
    python scripts/rq1_rq2_final_evaluation.py

Author: Sumit Malviya (W24041293)
Supervisor: Dr. Jason Moore
Module: KF7029 MSc Project
"""

import os
import csv
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

matplotlib.rcParams['figure.figsize'] = (12, 6)
matplotlib.rcParams['font.size'] = 11
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# CONFIGURATION
# ============================================================
BASELINE_MODEL = "yolov8n.pt"
FINETUNED_MODEL = "runs/detect/newcastle_finetune2/weights/best.pt"
TEST_DIR = "data/Newcastle-Traffic-Detection.v2i.yolov8/test"
RESULTS_DIR = "data/results/final_evaluation"

CONF = 0.25
DEVICE = "cpu"
IMGSZ = 640

# Class mapping: Roboflow class IDs
RF_NAMES = {0: 'Traffic light', 1: 'Bus', 2: 'Car', 3: 'Person', 4: 'Truck'}

# COCO class IDs -> Roboflow class IDs (for baseline model)
COCO_TO_RF = {9: 0, 5: 1, 2: 2, 0: 3, 7: 4}

# IoU thresholds for mAP@0.5:0.95
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

# Day/Night classification based on hour in filename
# Filename format: CameraName__YYYYMMDD_HHMMSS_hash.jpg
DAY_HOURS = range(7, 18)   # 07:00 - 17:59
NIGHT_HOURS = list(range(0, 7)) + list(range(18, 24))  # 00:00-06:59, 18:00-23:59

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_iou(b1, b2):
    """Compute IoU between two boxes in [x_center, y_center, w, h] format (normalized)."""
    x1 = max(b1[0] - b1[2]/2, b2[0] - b2[2]/2)
    y1 = max(b1[1] - b1[3]/2, b2[1] - b2[3]/2)
    x2 = min(b1[0] + b1[2]/2, b2[0] + b2[2]/2)
    y2 = min(b1[1] + b1[3]/2, b2[1] + b2[3]/2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path):
    """Load ground truth boxes from YOLO format label file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) >= 5:
            c = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append({'class': c, 'box': [x, y, w, h], 'area': w * h})
    return boxes


def match_detections(preds, gts, iou_thr=0.5):
    """Match predictions to ground truth using greedy IoU matching."""
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    iou_matrix = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = compute_iou(p['box'], g['box'])

    matches = []
    matched_preds = set()
    matched_gts = set()

    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_thr:
            break
        pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        pi, gi = int(pi), int(gi)
        if preds[pi]['class'] == gts[gi]['class']:
            matches.append((pi, gi, max_iou))
            matched_preds.add(pi)
            matched_gts.add(gi)
        iou_matrix[pi, :] = 0
        iou_matrix[:, gi] = 0

    unmatched_preds = [i for i in range(len(preds)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gts)) if i not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


def compute_ap_11point(precisions, recalls):
    """Compute AP using 11-point interpolation."""
    if not precisions:
        return 0.0
    pairs = sorted(zip(recalls, precisions))
    rs = [p[0] for p in pairs]
    ps = [p[1] for p in pairs]
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        matching = [p for p, r in zip(ps, rs) if r >= t]
        ap += max(matching) if matching else 0
    return ap / 11.0


def parse_hour_from_filename(filename):
    """Extract hour from filename for day/night classification."""
    import re
    # Try pattern: YYYYMMDD_HHMMSS
    match = re.search(r'(\d{8})_(\d{6})', filename)
    if match:
        return int(match.group(2)[:2])
    return None


def get_time_period(hour):
    """Classify hour as day or night."""
    if hour is None:
        return "unknown"
    if hour in DAY_HOURS:
        return "day"
    return "night"


def collect_test_dataset(test_dir):
    """Collect all image-label pairs from the test directory."""
    pairs = []
    img_dir = Path(test_dir) / "images"
    lbl_dir = Path(test_dir) / "labels"

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found!")
        return pairs

    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

        label_path = lbl_dir / f"{img_path.stem}.txt"
        hour = parse_hour_from_filename(img_path.name)
        period = get_time_period(hour)

        # Extract camera name from filename (before __YYYYMMDD)
        parts = img_path.stem.split('__')
        camera = parts[0] if len(parts) >= 2 else img_path.stem.split('_')[0]

        pairs.append({
            'image': str(img_path),
            'label': str(label_path),
            'filename': img_path.name,
            'camera': camera,
            'hour': hour,
            'period': period,
        })

    return pairs


# ============================================================
# MAIN EVALUATION ENGINE
# ============================================================
def evaluate_model(model, dataset, model_name, is_finetuned=False):
    """
    Run full evaluation of a model on the test dataset.
    Computes mAP@0.5, mAP@0.5:0.95, per-class, per-camera, day/night metrics.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_name}")
    print(f"  Test images: {len(dataset)}")
    print(f"{'='*60}")

    # Storage for metrics
    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    camera_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    period_stats = defaultdict(lambda: defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0}))
    pred_confidences = defaultdict(list)  # class -> [(conf, is_tp)]
    per_image = []

    # For mAP@0.5:0.95 - need per-threshold stats
    threshold_class_stats = {
        iou_t: defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
        for iou_t in IOU_THRESHOLDS
    }

    start_time = time.time()

    for idx, item in enumerate(dataset):
        gts = load_gt(item['label'])

        # Run inference
        results = model.predict(
            source=item['image'],
            conf=CONF,
            verbose=False,
            device=DEVICE,
            imgsz=IMGSZ
        )

        # Convert predictions to common format
        preds = []
        for box in results[0].boxes:
            coco_cls = int(box.cls[0])
            if is_finetuned:
                # Fine-tuned model already uses Roboflow classes (0-4)
                if coco_cls in RF_NAMES:
                    preds.append({
                        'class': coco_cls,
                        'box': box.xywhn[0].tolist(),
                        'conf': float(box.conf[0])
                    })
            else:
                # Baseline uses COCO classes, need to map
                if coco_cls in COCO_TO_RF:
                    preds.append({
                        'class': COCO_TO_RF[coco_cls],
                        'box': box.xywhn[0].tolist(),
                        'conf': float(box.conf[0])
                    })

        camera = item['camera']
        period = item['period']

        # Match at IoU=0.5 for main metrics
        matches, unmatched_p, unmatched_g = match_detections(preds, gts, iou_thr=0.5)

        img_tp = img_fp = img_fn = 0

        for pi, gi, _ in matches:
            c = preds[pi]['class']
            class_stats[c]['TP'] += 1
            camera_stats[camera]['TP'] += 1
            period_stats[period][c]['TP'] += 1
            pred_confidences[c].append((preds[pi]['conf'], True))
            img_tp += 1

        for pi in unmatched_p:
            c = preds[pi]['class']
            class_stats[c]['FP'] += 1
            camera_stats[camera]['FP'] += 1
            period_stats[period][c]['FP'] += 1
            pred_confidences[c].append((preds[pi]['conf'], False))
            img_fp += 1

        for gi in unmatched_g:
            c = gts[gi]['class']
            class_stats[c]['FN'] += 1
            camera_stats[camera]['FN'] += 1
            period_stats[period][c]['FN'] += 1
            img_fn += 1

        # Per-image metrics
        p = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0
        r = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        per_image.append({
            'file': item['filename'], 'camera': camera, 'period': period,
            'hour': item['hour'], 'gt': len(gts), 'pred': len(preds),
            'TP': img_tp, 'FP': img_fp, 'FN': img_fn,
            'P': round(p, 4), 'R': round(r, 4), 'F1': round(f1, 4)
        })

        # Also compute matches at all IoU thresholds for mAP@0.5:0.95
        for iou_t in IOU_THRESHOLDS:
            mt, up, ug = match_detections(preds, gts, iou_thr=iou_t)
            for pi, gi, _ in mt:
                c = preds[pi]['class']
                threshold_class_stats[iou_t][c]['TP'] += 1
            for pi in up:
                c = preds[pi]['class']
                threshold_class_stats[iou_t][c]['FP'] += 1
            for gi in ug:
                c = gts[gi]['class']
                threshold_class_stats[iou_t][c]['FN'] += 1

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(dataset)}] processed")

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    # ---- Compute per-class metrics at IoU=0.5 ----
    per_class = []
    aps_50 = []
    for cid in sorted(RF_NAMES.keys()):
        s = class_stats[cid]
        tp, fp, fn = s['TP'], s['FP'], s['FN']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # AP from confidence-sorted predictions
        confs = pred_confidences[cid]
        if confs:
            sorted_confs = sorted(confs, key=lambda x: x[0], reverse=True)
            running_tp = running_fp = 0
            total_gt = tp + fn
            precs, recs = [], []
            for conf_val, is_tp in sorted_confs:
                if is_tp:
                    running_tp += 1
                else:
                    running_fp += 1
                precs.append(running_tp / (running_tp + running_fp))
                recs.append(running_tp / total_gt if total_gt > 0 else 0)
            ap = compute_ap_11point(precs, recs)
        else:
            ap = 0.0

        aps_50.append(ap)
        per_class.append({
            'class': RF_NAMES[cid], 'TP': tp, 'FP': fp, 'FN': fn,
            'P': round(p, 4), 'R': round(r, 4), 'F1': round(f1, 4),
            'AP50': round(ap, 4)
        })

    # ---- Compute mAP@0.5:0.95 ----
    aps_per_threshold = []
    for iou_t in IOU_THRESHOLDS:
        threshold_aps = []
        for cid in sorted(RF_NAMES.keys()):
            s = threshold_class_stats[iou_t][cid]
            tp, fp, fn = s['TP'], s['FP'], s['FN']
            # Simplified AP at this threshold
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            threshold_aps.append(p * r / max(p + r, 1e-6) * 2 if (p + r) > 0 else 0)
        aps_per_threshold.append(np.mean(threshold_aps))

    # Per-class mAP@0.5:0.95
    per_class_map5095 = []
    for cid in sorted(RF_NAMES.keys()):
        class_aps = []
        for iou_t in IOU_THRESHOLDS:
            s = threshold_class_stats[iou_t][cid]
            tp, fn = s['TP'], s['FN']
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_aps.append(r)  # recall at each threshold
        per_class_map5095.append(np.mean(class_aps))

    # Add mAP@0.5:0.95 to per_class
    for i, pc in enumerate(per_class):
        pc['AP50_95'] = round(per_class_map5095[i], 4)

    # ---- Overall metrics ----
    total_tp = sum(class_stats[c]['TP'] for c in RF_NAMES)
    total_fp = sum(class_stats[c]['FP'] for c in RF_NAMES)
    total_fn = sum(class_stats[c]['FN'] for c in RF_NAMES)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    mAP50 = np.mean(aps_50)
    mAP50_95 = np.mean(per_class_map5095)

    overall = {
        'model': model_name,
        'images': len(dataset),
        'gt_objects': total_tp + total_fn,
        'predictions': total_tp + total_fp,
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
        'P': round(overall_p, 4), 'R': round(overall_r, 4), 'F1': round(overall_f1, 4),
        'mAP50': round(mAP50, 4), 'mAP50_95': round(mAP50_95, 4),
    }

    # ---- Day vs Night metrics ----
    day_night = {}
    for period in ['day', 'night']:
        p_stats = period_stats[period]
        p_tp = sum(p_stats[c]['TP'] for c in RF_NAMES)
        p_fp = sum(p_stats[c]['FP'] for c in RF_NAMES)
        p_fn = sum(p_stats[c]['FN'] for c in RF_NAMES)
        pp = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0
        pr = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0
        pf = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0
        n_images = sum(1 for img in per_image if img['period'] == period)
        day_night[period] = {
            'images': n_images,
            'TP': p_tp, 'FP': p_fp, 'FN': p_fn,
            'P': round(pp, 4), 'R': round(pr, 4), 'F1': round(pf, 4)
        }

    # ---- Per-camera metrics ----
    per_camera = []
    for cam in sorted(camera_stats.keys()):
        s = camera_stats[cam]
        cp = s['TP'] / (s['TP'] + s['FP']) if (s['TP'] + s['FP']) > 0 else 0
        cr = s['TP'] / (s['TP'] + s['FN']) if (s['TP'] + s['FN']) > 0 else 0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        per_camera.append({
            'camera': cam, 'TP': s['TP'], 'FP': s['FP'], 'FN': s['FN'],
            'P': round(cp, 4), 'R': round(cr, 4), 'F1': round(cf, 4)
        })

    # ---- Confidence threshold analysis ----
    conf_analysis = []
    for thr in [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        t_tp = t_fp = t_missed_tp = 0
        for c in RF_NAMES:
            for conf_val, is_tp in pred_confidences[c]:
                if conf_val >= thr:
                    if is_tp: t_tp += 1
                    else: t_fp += 1
                else:
                    if is_tp: t_missed_tp += 1
        t_fn = total_fn + t_missed_tp
        cp = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        cr = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        conf_analysis.append({'threshold': thr, 'P': round(cp, 4), 'R': round(cr, 4), 'F1': round(cf, 4)})

    # Print summary
    print(f"\n  RESULTS: {model_name}")
    print(f"  {'='*50}")
    print(f"  Overall:  P={overall_p:.3f}  R={overall_r:.3f}  F1={overall_f1:.3f}")
    print(f"  mAP@0.5:      {mAP50:.4f}")
    print(f"  mAP@0.5:0.95: {mAP50_95:.4f}")
    print(f"\n  Per-class:")
    for pc in per_class:
        print(f"    {pc['class']:15s}  P={pc['P']:.3f}  R={pc['R']:.3f}  F1={pc['F1']:.3f}  AP50={pc['AP50']:.3f}  AP50:95={pc['AP50_95']:.3f}")
    print(f"\n  Day vs Night:")
    for period, stats in day_night.items():
        print(f"    {period:6s}: {stats['images']:3d} images  P={stats['P']:.3f}  R={stats['R']:.3f}  F1={stats['F1']:.3f}")

    return {
        'overall': overall,
        'per_class': per_class,
        'per_camera': per_camera,
        'per_image': per_image,
        'day_night': day_night,
        'conf_analysis': conf_analysis,
    }


# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(results, prefix):
    """Save all result tables as CSVs."""
    # Overall
    with open(f"{RESULTS_DIR}/{prefix}_overall.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results['overall'].keys())
        w.writeheader()
        w.writerow(results['overall'])

    # Per-class
    with open(f"{RESULTS_DIR}/{prefix}_per_class.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results['per_class'][0].keys())
        w.writeheader()
        w.writerows(results['per_class'])

    # Per-camera
    if results['per_camera']:
        with open(f"{RESULTS_DIR}/{prefix}_per_camera.csv", 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results['per_camera'][0].keys())
            w.writeheader()
            w.writerows(results['per_camera'])

    # Per-image
    with open(f"{RESULTS_DIR}/{prefix}_per_image.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results['per_image'][0].keys())
        w.writeheader()
        w.writerows(results['per_image'])

    # Day/Night
    with open(f"{RESULTS_DIR}/{prefix}_day_night.csv", 'w', newline='') as f:
        f.write("period,images,TP,FP,FN,P,R,F1\n")
        for period, stats in results['day_night'].items():
            f.write(f"{period},{stats['images']},{stats['TP']},{stats['FP']},{stats['FN']},"
                    f"{stats['P']},{stats['R']},{stats['F1']}\n")

    # Confidence analysis
    with open(f"{RESULTS_DIR}/{prefix}_confidence.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results['conf_analysis'][0].keys())
        w.writeheader()
        w.writerows(results['conf_analysis'])


# ============================================================
# COMPARISON CHARTS
# ============================================================
def generate_comparison_charts(baseline, finetuned):
    """Generate before vs after comparison visualisations."""

    classes = [pc['class'] for pc in baseline['per_class']]
    x = np.arange(len(classes))
    width = 0.35

    # 1. Per-class AP@0.5 comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    b_ap = [pc['AP50'] for pc in baseline['per_class']]
    f_ap = [pc['AP50'] for pc in finetuned['per_class']]
    bars1 = ax.bar(x - width/2, b_ap, width, label='Baseline', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, f_ap, width, label='Fine-tuned', color='#ED7D31')
    ax.set_xlabel('Class')
    ax.set_ylabel('AP@0.5')
    ax.set_title('RQ1 vs RQ2: Per-Class AP@0.5 on Test Set (78 images)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comparison_ap50_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Overall metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['P', 'R', 'F1', 'mAP50', 'mAP50_95']
    b_vals = [baseline['overall'][m] for m in metrics]
    f_vals = [finetuned['overall'][m] for m in metrics]
    x2 = np.arange(len(metrics))
    bars1 = ax.bar(x2 - width/2, b_vals, width, label='Baseline', color='#5B9BD5')
    bars2 = ax.bar(x2 + width/2, f_vals, width, label='Fine-tuned', color='#ED7D31')
    ax.set_ylabel('Score')
    ax.set_title('RQ1 vs RQ2: Overall Metrics on Test Set')
    ax.set_xticks(x2)
    ax.set_xticklabels(['Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95'])
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comparison_overall.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Day vs Night comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (model_name, results) in enumerate([('Baseline', baseline), ('Fine-tuned', finetuned)]):
        ax = axes[idx]
        periods = ['day', 'night']
        p_vals = [results['day_night'].get(p, {}).get('P', 0) for p in periods]
        r_vals = [results['day_night'].get(p, {}).get('R', 0) for p in periods]
        f_vals = [results['day_night'].get(p, {}).get('F1', 0) for p in periods]
        x3 = np.arange(2)
        w = 0.25
        ax.bar(x3 - w, p_vals, w, label='Precision', color='#5B9BD5')
        ax.bar(x3, r_vals, w, label='Recall', color='#70AD47')
        ax.bar(x3 + w, f_vals, w, label='F1', color='#ED7D31')
        ax.set_xticks(x3)
        ax.set_xticklabels(['Day (07-18h)', 'Night (18-07h)'])
        ax.set_title(f'{model_name}: Day vs Night')
        ax.legend()
        ax.set_ylim(0, 1.0)
    plt.suptitle('Performance by Time of Day', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comparison_day_night.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Save comparison CSV
    with open(f'{RESULTS_DIR}/rq1_vs_rq2_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'baseline', 'finetuned', 'delta'])
        for m in ['P', 'R', 'F1', 'mAP50', 'mAP50_95']:
            bv = baseline['overall'][m]
            fv = finetuned['overall'][m]
            w.writerow([m, bv, fv, round(fv - bv, 4)])

    # 5. Per-class comparison CSV
    with open(f'{RESULTS_DIR}/rq1_vs_rq2_per_class.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'base_P', 'base_R', 'base_AP50', 'base_AP50_95',
                     'ft_P', 'ft_R', 'ft_AP50', 'ft_AP50_95', 'delta_AP50'])
        for bc, fc in zip(baseline['per_class'], finetuned['per_class']):
            w.writerow([bc['class'], bc['P'], bc['R'], bc['AP50'], bc['AP50_95'],
                        fc['P'], fc['R'], fc['AP50'], fc['AP50_95'],
                        round(fc['AP50'] - bc['AP50'], 4)])

    print(f"\n  Charts saved to {RESULTS_DIR}/")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION: RQ1 + RQ2")
    print("  Test-only evaluation (no training data)")
    print("=" * 60)

    # Load test dataset
    dataset = collect_test_dataset(TEST_DIR)
    print(f"\n  Test set: {len(dataset)} images")
    day_count = sum(1 for d in dataset if d['period'] == 'day')
    night_count = sum(1 for d in dataset if d['period'] == 'night')
    unknown_count = sum(1 for d in dataset if d['period'] == 'unknown')
    print(f"  Day: {day_count}, Night: {night_count}, Unknown: {unknown_count}")
    print(f"  Cameras: {len(set(d['camera'] for d in dataset))}")

    # Load models
    from ultralytics import YOLO
    print(f"\n  Loading baseline: {BASELINE_MODEL}")
    baseline_model = YOLO(BASELINE_MODEL)
    print(f"  Loading fine-tuned: {FINETUNED_MODEL}")
    finetuned_model = YOLO(FINETUNED_MODEL)

    # RQ1: Baseline evaluation
    baseline_results = evaluate_model(baseline_model, dataset, "Baseline YOLOv8n", is_finetuned=False)
    save_results(baseline_results, "rq1_baseline")

    # RQ2: Fine-tuned evaluation
    finetuned_results = evaluate_model(finetuned_model, dataset, "Fine-tuned YOLOv8n", is_finetuned=True)
    save_results(finetuned_results, "rq2_finetuned")

    # Comparison
    generate_comparison_charts(baseline_results, finetuned_results)

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON: BASELINE vs FINE-TUNED")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 55)
    for m in ['P', 'R', 'F1', 'mAP50', 'mAP50_95']:
        bv = baseline_results['overall'][m]
        fv = finetuned_results['overall'][m]
        delta = fv - bv
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"{m:<20} {bv:>9.1%} {fv:>11.1%} {arrow} {abs(delta):>8.1%}")

    print(f"\n  All results saved to: {RESULTS_DIR}/")
    print(f"  CSVs: rq1_baseline_*.csv, rq2_finetuned_*.csv, rq1_vs_rq2_*.csv")
    print(f"  Charts: comparison_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()