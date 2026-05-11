"""
RQ1 + RQ2 Final Evaluation — FIXED VERSION
============================================
Fixes applied:
  1. Greedy matching: only zero row/col on valid class-consistent match
  2. mAP@0.5:0.95: proper AP computation at each IoU threshold
  3. Per-class AP@0.5:0.95 output
  4. IMGSZ=960 to match training
  5. Proper confusion matrix (class-to-class)
  6. Full CSV export (overall, day/night, per-camera, per-class AP50:95)
  7. Correct dataset numbers (344 test, 44 cameras)

Author: Sumit Malviya (W24041293)
"""

import os, re, time, csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
BASELINE_MODEL_PATH = "yolov8n.pt"
FINETUNED_MODEL_PATH = "runs/detect/newcastle_v6_improved/weights/best.pt"
DATASET_DIR = Path("data/Newcastle-Traffic-Detection.v6i.yolov8")
TEST_DIR = DATASET_DIR / "test"
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.25
DEVICE = "cpu"
IMGSZ = 960  # FIX #4: match training resolution

RF_NAMES = {0:"Motorcycle", 1:"Roadwork", 2:"Traffic light", 3:"bicycle",
            4:"bus", 5:"car", 6:"person", 7:"truck"}
COCO_TO_RF = {3:0, 9:2, 1:3, 5:4, 2:5, 0:6, 7:7}
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
NUM_CLASSES = len(RF_NAMES)


# ============================================================
# FIX #1: Correct greedy matching — only zero on class match
# ============================================================
def compute_iou(b1, b2):
    x1 = max(b1[0]-b1[2]/2, b2[0]-b2[2]/2)
    y1 = max(b1[1]-b1[3]/2, b2[1]-b2[3]/2)
    x2 = min(b1[0]+b1[2]/2, b2[0]+b2[2]/2)
    y2 = min(b1[1]+b1[3]/2, b2[1]+b2[3]/2)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0


def greedy_match_fixed(preds, gts, iou_thr=0.5):
    """
    FIX #1: Only zero out row/column when class-consistent match is accepted.
    Wrong-class high-IoU pairs no longer block correct later matches.
    """
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    M = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            # Only compute IoU for same-class pairs
            if p["class"] == g["class"]:
                M[i, j] = compute_iou(p["box"], g["box"])

    matches, matched_p, matched_g = [], set(), set()
    while True:
        if M.size == 0:
            break
        mx = M.max()
        if mx < iou_thr:
            break
        pi, gi = np.unravel_index(M.argmax(), M.shape)
        pi, gi = int(pi), int(gi)
        # Class already matches (we only filled same-class IoU)
        matches.append((pi, gi, mx))
        matched_p.add(pi)
        matched_g.add(gi)
        M[pi, :] = 0
        M[:, gi] = 0

    unmatched_p = [i for i in range(len(preds)) if i not in matched_p]
    unmatched_g = [i for i in range(len(gts)) if i not in matched_g]
    return matches, unmatched_p, unmatched_g


# ============================================================
# FIX #2: Correct AP computation (11-point interpolation)
# ============================================================
def compute_ap_11point(confidences_and_tps, total_gt):
    """
    Given list of (confidence, is_tp) sorted by confidence desc,
    compute AP using 11-point interpolation.
    """
    if not confidences_and_tps or total_gt == 0:
        return 0.0

    sorted_dets = sorted(confidences_and_tps, key=lambda x: x[0], reverse=True)
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for conf, is_tp in sorted_dets:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_r) if p_at_r else 0
    return ap / 11.0


# ============================================================
# HELPERS
# ============================================================
def load_gt(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append({"class": int(parts[0]),
                          "box": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]})
    return boxes


def parse_hour(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        h = int(m.group(2)[:2])
        return h if 0 <= h <= 23 else None
    return None


# ============================================================
# MAIN EVALUATION ENGINE
# ============================================================
def evaluate_model(model, test_data, model_name, is_baseline=False):
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Test images: {len(test_data)}  |  IMGSZ: {IMGSZ}  |  Conf: {CONF}")
    print(f"{'='*70}")

    # Storage
    class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    period_stats = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))
    camera_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "GT": 0})
    per_class_confs = defaultdict(list)  # class -> [(conf, is_tp)] at IoU=0.5

    # FIX #2: per-class confidence lists at EACH IoU threshold
    per_class_confs_strict = {
        round(t, 2): defaultdict(list) for t in IOU_THRESHOLDS
    }

    # FIX #5: confusion matrix (class-to-class + background)
    # conf_matrix[pred_class][gt_class] counts
    conf_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1))  # +1 for background

    t0 = time.time()

    for idx, item in enumerate(test_data):
        gts = load_gt(item["label"])
        results = model.predict(source=item["image"], conf=CONF, verbose=False,
                                device=DEVICE, imgsz=IMGSZ)

        preds = []
        for b in results[0].boxes:
            cid = int(b.cls[0])
            conf = float(b.conf[0])
            box = b.xywhn[0].tolist()
            if is_baseline:
                if cid in COCO_TO_RF:
                    preds.append({"class": COCO_TO_RF[cid], "box": box, "conf": conf})
            else:
                if cid in RF_NAMES:
                    preds.append({"class": cid, "box": box, "conf": conf})

        # --- IoU=0.5 matching (main metrics) ---
        matches, unmatched_p, unmatched_g = greedy_match_fixed(preds, gts, 0.5)

        for pi, gi, iou in matches:
            c = preds[pi]["class"]
            class_stats[c]["TP"] += 1
            period_stats[item["period"]][c]["TP"] += 1
            camera_stats[item["camera"]]["TP"] += 1
            per_class_confs[c].append((preds[pi]["conf"], True))
            conf_matrix[c][c] += 1  # correct prediction

        for pi in unmatched_p:
            c = preds[pi]["class"]
            class_stats[c]["FP"] += 1
            period_stats[item["period"]][c]["FP"] += 1
            camera_stats[item["camera"]]["FP"] += 1
            per_class_confs[c].append((preds[pi]["conf"], False))
            # FIX #5: FP goes to pred_class row, background column
            conf_matrix[c][NUM_CLASSES] += 1

        for gi in unmatched_g:
            c = gts[gi]["class"]
            class_stats[c]["FN"] += 1
            period_stats[item["period"]][c]["FN"] += 1
            camera_stats[item["camera"]]["FN"] += 1
            # FIX #5: FN goes to background row, gt_class column
            conf_matrix[NUM_CLASSES][c] += 1

        for g in gts:
            camera_stats[item["camera"]]["GT"] += 1

        # --- FIX #5: Build real class-to-class confusion ---
        # For FP with wrong class, find closest GT and record confusion
        # (simplified: check all unmatched preds against all unmatched GTs)
        for pi in unmatched_p:
            pred_c = preds[pi]["class"]
            best_iou = 0
            best_gt_c = None
            for gi in unmatched_g:
                iou = compute_iou(preds[pi]["box"], gts[gi]["box"])
                if iou > best_iou and iou >= 0.3:  # loose threshold for confusion
                    best_iou = iou
                    best_gt_c = gts[gi]["class"]
            if best_gt_c is not None and best_gt_c != pred_c:
                # Class confusion: predicted as pred_c, was actually best_gt_c
                conf_matrix[pred_c][best_gt_c] += 1
                conf_matrix[pred_c][NUM_CLASSES] -= 1  # remove from background column

        # --- Multi-threshold matching for mAP@0.5:0.95 ---
        for iou_thr in IOU_THRESHOLDS:
            k = round(iou_thr, 2)
            mt, up, ug = greedy_match_fixed(preds, gts, iou_thr)
            matched_pred_set = set()
            for pi, gi, _ in mt:
                c = preds[pi]["class"]
                per_class_confs_strict[k][c].append((preds[pi]["conf"], True))
                matched_pred_set.add(pi)
            for pi in up:
                c = preds[pi]["class"]
                per_class_confs_strict[k][c].append((preds[pi]["conf"], False))

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(test_data)}]")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/len(test_data):.2f}s/img)")

    # ================================================================
    # COMPUTE PER-CLASS METRICS
    # ================================================================
    per_class = []
    all_aps_50 = []

    # FIX #2 + #3: Compute proper AP at each IoU threshold per class
    per_class_ap50_95 = []

    for cid in sorted(RF_NAMES.keys()):
        s = class_stats[cid]
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        gt = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # AP@0.5
        ap50 = compute_ap_11point(per_class_confs[cid], gt)
        all_aps_50.append(ap50)

        # FIX #2: Proper AP at each IoU threshold, then average
        aps_at_thresholds = []
        for iou_thr in IOU_THRESHOLDS:
            k = round(iou_thr, 2)
            class_gt = gt  # same GT count regardless of IoU threshold
            ap_at_t = compute_ap_11point(per_class_confs_strict[k][cid], class_gt)
            aps_at_thresholds.append(ap_at_t)
        ap50_95 = np.mean(aps_at_thresholds)
        per_class_ap50_95.append(ap50_95)

        per_class.append({
            "cid": cid, "name": RF_NAMES[cid], "GT": gt,
            "TP": tp, "FP": fp, "FN": fn,
            "P": p, "R": r, "F1": f1,
            "AP50": ap50, "AP50_95": ap50_95   # FIX #3: include AP50:95
        })

    mAP50 = np.mean(all_aps_50)
    mAP50_95 = np.mean(per_class_ap50_95)

    # Overall
    ttp = sum(class_stats[c]["TP"] for c in RF_NAMES)
    tfp = sum(class_stats[c]["FP"] for c in RF_NAMES)
    tfn = sum(class_stats[c]["FN"] for c in RF_NAMES)
    op = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0
    orr = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0
    of1 = 2 * op * orr / (op + orr) if (op + orr) > 0 else 0

    # Day/Night
    day_night = {}
    for period in ["day", "night"]:
        ps = period_stats[period]
        ptp = sum(ps[c]["TP"] for c in RF_NAMES)
        pfp = sum(ps[c]["FP"] for c in RF_NAMES)
        pfn = sum(ps[c]["FN"] for c in RF_NAMES)
        pp = ptp / (ptp + pfp) if (ptp + pfp) > 0 else 0
        pr = ptp / (ptp + pfn) if (ptp + pfn) > 0 else 0
        pf = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0
        ni = sum(1 for d in test_data if d["period"] == period)
        day_night[period] = {"imgs": ni, "P": pp, "R": pr, "F1": pf}

    # Per-camera
    per_camera = []
    for cam in sorted(camera_stats.keys(), key=lambda c: camera_stats[c]["GT"], reverse=True):
        s = camera_stats[cam]
        cp = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0
        cr = s["TP"] / s["GT"] if s["GT"] > 0 else 0
        per_camera.append({"camera": cam, "GT": s["GT"], "P": cp, "R": cr})

    # ================================================================
    # PRINT RESULTS
    # ================================================================
    print(f"\n  Per-class results (IoU=0.5):")
    header = f"  {'Class':18s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AP@0.5':>8s} {'AP@.5:.95':>10s}"
    print(header)
    print(f"  {'-'*80}")
    for pc in per_class:
        name=pc['name']; gt=pc['GT']; tp=pc['TP']; fp=pc['FP']; fn=pc['FN']
        p=pc['P']; r=pc['R']; f1=pc['F1']; ap50=pc['AP50']; ap5095=pc['AP50_95']
        print(f"  {name:18s} {gt:5d} {tp:5d} {fp:5d} {fn:5d} {p:6.1%} {r:6.1%} {f1:6.1%} {ap50:7.1%} {ap5095:9.1%}")

    tgt = ttp + tfn
    print(f"\n  {'OVERALL':18s} {tgt:5d} {ttp:5d} {tfp:5d} {tfn:5d} {op:6.1%} {orr:6.1%} {of1:6.1%} {mAP50:7.1%} {mAP50_95:9.1%}")
    print(f"\n  mAP@0.5      = {mAP50:.4f} ({mAP50:.1%})")
    print(f"  mAP@0.5:0.95 = {mAP50_95:.4f} ({mAP50_95:.1%})")

    print(f"\n  Day vs Night:")
    for period in ["day", "night"]:
        s = day_night[period]
        imgs=s['imgs']; pp=s['P']; rr=s['R']; ff=s['F1']
        print(f"    {period:6s}: {imgs:3d} imgs  P={pp:.1%}  R={rr:.1%}  F1={ff:.1%}")

    print(f"\n  Per-camera (top 5):")
    for c in per_camera[:5]:
        cam=c['camera']; gt=c['GT']; cp=c['P']; cr=c['R']
        print(f"    {cam[:40]:40s}  GT={gt:4d}  P={cp:.1%}  R={cr:.1%}")

    return {
        "model": model_name,
        "overall": {"P": op, "R": orr, "F1": of1, "mAP50": mAP50, "mAP50_95": mAP50_95},
        "per_class": per_class,
        "day_night": day_night,
        "per_camera": per_camera,
        "confusion_matrix": conf_matrix,
    }


# ============================================================
# FIX #6: Full CSV export
# ============================================================
def save_full_results(rq1, rq2):
    """Save comprehensive CSVs for all metrics."""

    # 1. Overall comparison
    with open(RESULTS_DIR / "overall_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "finetuned", "delta"])
        for key in ["P", "R", "F1", "mAP50", "mAP50_95"]:
            bv = rq1["overall"][key]
            fv = rq2["overall"][key]
            w.writerow([key, round(bv, 4), round(fv, 4), round(fv - bv, 4)])

    # 2. Per-class with AP50 AND AP50:95 (FIX #3)
    with open(RESULTS_DIR / "per_class_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "GT",
                     "bl_P", "bl_R", "bl_F1", "bl_AP50", "bl_AP50_95",
                     "ft_P", "ft_R", "ft_F1", "ft_AP50", "ft_AP50_95",
                     "delta_AP50", "delta_AP50_95"])
        for bp, fp in zip(rq1["per_class"], rq2["per_class"]):
            w.writerow([bp["name"], bp["GT"],
                        round(bp["P"],4), round(bp["R"],4), round(bp["F1"],4),
                        round(bp["AP50"],4), round(bp["AP50_95"],4),
                        round(fp["P"],4), round(fp["R"],4), round(fp["F1"],4),
                        round(fp["AP50"],4), round(fp["AP50_95"],4),
                        round(fp["AP50"]-bp["AP50"],4),
                        round(fp["AP50_95"]-bp["AP50_95"],4)])

    # 3. Day/Night
    with open(RESULTS_DIR / "day_night_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "bl_imgs", "bl_P", "bl_R", "bl_F1",
                     "ft_imgs", "ft_P", "ft_R", "ft_F1"])
        for period in ["day", "night"]:
            b = rq1["day_night"][period]
            ft = rq2["day_night"][period]
            w.writerow([period, b["imgs"], round(b["P"],4), round(b["R"],4), round(b["F1"],4),
                        ft["imgs"], round(ft["P"],4), round(ft["R"],4), round(ft["F1"],4)])

    # 4. Per-camera
    with open(RESULTS_DIR / "per_camera_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "GT", "bl_P", "bl_R", "ft_P", "ft_R"])
        bl_cams = {c["camera"]: c for c in rq1["per_camera"]}
        for fc in rq2["per_camera"]:
            cam = fc["camera"]
            bc = bl_cams.get(cam, {"P": 0, "R": 0})
            w.writerow([cam, fc["GT"],
                        round(bc["P"],4), round(bc["R"],4),
                        round(fc["P"],4), round(fc["R"],4)])

    # 5. Confusion matrix
    labels = [RF_NAMES[i] for i in sorted(RF_NAMES.keys())] + ["Background"]
    np.savetxt(RESULTS_DIR / "confusion_matrix_finetuned.csv",
               rq2["confusion_matrix"], delimiter=",", fmt="%.0f",
               header=",".join(labels), comments="")

    print(f"\n  All CSVs saved to {RESULTS_DIR}/")


# ============================================================
# LOAD TEST DATA
# ============================================================
print("\n" + "=" * 70)
print("  LOADING TEST DATASET")
print("=" * 70)

test_data = []
img_dir = TEST_DIR / "images"
lbl_dir = TEST_DIR / "labels"

for img_path in sorted(img_dir.glob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    parts = img_path.stem.split("__")
    camera = parts[0] if len(parts) >= 2 else img_path.stem.split("_")[0]
    hour = parse_hour(img_path.name)
    period = "day" if hour is not None and 7 <= hour < 18 else "night"
    test_data.append({
        "image": str(img_path), "label": str(lbl_dir / f"{img_path.stem}.txt"),
        "filename": img_path.name, "camera": camera, "hour": hour, "period": period
    })

gt_counts = defaultdict(int)
for d in test_data:
    for gt in load_gt(d["label"]):
        gt_counts[gt["class"]] += 1

day_n = sum(1 for d in test_data if d["period"] == "day")
night_n = sum(1 for d in test_data if d["period"] == "night")
n_cams = len(set(d["camera"] for d in test_data))
total_gt = sum(gt_counts.values())

print(f"  Test images: {len(test_data)} ({day_n} day, {night_n} night)")
print(f"  Cameras:     {n_cams}")
print(f"\n  Ground truth:")
for cid in sorted(gt_counts):
    name = RF_NAMES.get(cid, "?")
    print(f"    {name:18s}: {gt_counts[cid]:5d}")
print(f"    {'TOTAL':18s}: {total_gt:5d}")


# ============================================================
# RUN EVALUATIONS
# ============================================================
baseline_model = YOLO(BASELINE_MODEL_PATH)
rq1 = evaluate_model(baseline_model, test_data, "Baseline YOLOv8n (COCO)", is_baseline=True)

finetuned_model = YOLO(FINETUNED_MODEL_PATH)
rq2 = evaluate_model(finetuned_model, test_data, "Fine-tuned YOLOv8s (v6, 789 imgs)", is_baseline=False)


# ============================================================
# COMPARISON (FIX #3: includes AP50:95)
# ============================================================
print(f"\n{'='*70}")
print(f"  COMPARISON: BASELINE vs FINE-TUNED")
print(f"{'='*70}")
header = f"  {'Metric':<18} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}"
print(header)
print(f"  {'-'*52}")
for key, label in [("P","Precision"),("R","Recall"),("F1","F1 Score"),
                    ("mAP50","mAP@0.5"),("mAP50_95","mAP@0.5:0.95")]:
    bv = rq1["overall"][key]
    fv = rq2["overall"][key]
    d = fv - bv
    sign = "+" if d > 0 else ""
    print(f"  {label:<18} {bv:>9.1%} {fv:>11.1%} {sign}{d:>8.1%}")

# FIX #3: Per-class with both AP50 AND AP50:95
print(f"\n  Per-class comparison:")
header2 = f"  {'Class':18s} {'bl_AP50':>8s} {'ft_AP50':>8s} {'d_AP50':>8s} {'bl_AP5095':>10s} {'ft_AP5095':>10s} {'d_AP5095':>10s}"
print(header2)
print(f"  {'-'*68}")
for bp, fp in zip(rq1["per_class"], rq2["per_class"]):
    name = bp["name"]
    b50 = bp["AP50"]; f50 = fp["AP50"]; d50 = f50 - b50
    b95 = bp["AP50_95"]; f95 = fp["AP50_95"]; d95 = f95 - b95
    s50 = "+" if d50 > 0 else ""
    s95 = "+" if d95 > 0 else ""
    print(f"  {name:18s} {b50:>7.1%} {f50:>7.1%} {s50}{d50:>7.1%} {b95:>9.1%} {f95:>9.1%} {s95}{d95:>9.1%}")


# ============================================================
# SAVE ALL RESULTS (FIX #6)
# ============================================================
save_full_results(rq1, rq2)

print(f"\n{'='*70}")
print(f"  DONE!")
print(f"  Fixes applied: matching bug, mAP@0.5:0.95, per-class AP50:95,")
print(f"  IMGSZ=960, confusion matrix, full CSV export")
print(f"{'='*70}")