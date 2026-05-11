"""
RQ1 + RQ2 Final Evaluation — VERSION 2 (4-Way Model Comparison)
================================================================
Changes from v1:
  - BASELINE is now YOLOv8s COCO (was YOLOv8n COCO)
  - Results saved to data/results_v2/ (original data/results/ untouched)
  - Output labelled clearly as YOLOv8s COCO vs YOLOv8s fine-tuned
  - Produces the 4-way comparison table for dissertation Section IV-B

Purpose:
  Isolates the contribution of domain adaptation from architecture size.
  Previous evaluation confounded both (YOLOv8n baseline vs YOLOv8s fine-tuned).
  This version holds architecture constant (both YOLOv8s) so the delta
  shows only the effect of fine-tuning on Newcastle CCTV imagery.

4-Way Table:
  Row 1: YOLOv8n COCO          → from original v1 results (overall_comparison.csv)
  Row 2: YOLOv8s COCO          → THIS RUN (baseline)
  Row 3: YOLOv8n fine-tuned    → optional (run rq2_finetune.py --model yolov8n.pt)
  Row 4: YOLOv8s fine-tuned    → THIS RUN (finetuned)

Author: Sumit Malviya (W24041293)
Supervisor: Dr. Jason Moore
Module: KF7029 — Northumbria University
"""

import os, re, time, csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# CONFIGURATION — only these two lines differ from v1
# ============================================================
BASELINE_MODEL_PATH  = "yolov8s.pt"                                          # CHANGED: was yolov8n.pt
FINETUNED_MODEL_PATH = "runs/detect/newcastle_v7/weights/best.pt"

DATASET_DIR = Path("data/Newcastle-Traffic-Detection.v6i.yolov8")
TEST_DIR    = DATASET_DIR / "test"
RESULTS_DIR = Path("data/results_v7")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF           = 0.25
DEVICE         = "cpu"
IMGSZ          = 960
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

RF_NAMES   = {0:"Motorcycle", 1:"Roadwork", 2:"Traffic light", 3:"bicycle",
              4:"bus", 5:"car", 6:"person", 7:"truck"}
COCO_TO_RF = {3:0, 9:2, 1:3, 5:4, 2:5, 0:6, 7:7}
NUM_CLASSES = len(RF_NAMES)


# ============================================================
# GREEDY MATCHING (same-class IoU only — fixed version)
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
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))
    M = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p["class"] == g["class"]:
                M[i, j] = compute_iou(p["box"], g["box"])
    matches, matched_p, matched_g = [], set(), set()
    while True:
        if M.size == 0: break
        mx = M.max()
        if mx < iou_thr: break
        pi, gi = np.unravel_index(M.argmax(), M.shape)
        pi, gi = int(pi), int(gi)
        matches.append((pi, gi, mx))
        matched_p.add(pi); matched_g.add(gi)
        M[pi, :] = 0; M[:, gi] = 0
    unmatched_p = [i for i in range(len(preds)) if i not in matched_p]
    unmatched_g = [i for i in range(len(gts))  if i not in matched_g]
    return matches, unmatched_p, unmatched_g


# ============================================================
# AP COMPUTATION (11-point interpolation)
# ============================================================
def compute_ap_11point(confidences_and_tps, total_gt):
    if not confidences_and_tps or total_gt == 0:
        return 0.0
    sorted_dets = sorted(confidences_and_tps, key=lambda x: x[0], reverse=True)
    tp_cumsum = fp_cumsum = 0
    precisions, recalls = [], []
    for conf, is_tp in sorted_dets:
        if is_tp: tp_cumsum += 1
        else:     fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt)
    ap = sum(max([p for p, r in zip(precisions, recalls) if r >= t], default=0)
             for t in np.arange(0, 1.1, 0.1))
    return ap / 11.0


# ============================================================
# HELPERS
# ============================================================
def load_gt(label_path):
    boxes = []
    if not os.path.exists(label_path): return boxes
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append({"class": int(parts[0]),
                          "box": [float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4])]})
    return boxes


def parse_hour(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        h = int(m.group(2)[:2])
        return h if 0 <= h <= 23 else None
    return None


# ============================================================
# EVALUATION ENGINE (identical to v1)
# ============================================================
def evaluate_model(model, test_data, model_name, is_baseline=False):
    print(f"\n{'='*70}")
    print(f"  Evaluating: {model_name}")
    print(f"  Test images: {len(test_data)}  |  IMGSZ: {IMGSZ}  |  Conf: {CONF}")
    print(f"{'='*70}")

    class_stats  = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    period_stats = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))
    camera_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "GT": 0})
    per_class_confs = defaultdict(list)
    per_class_confs_strict = {round(t, 2): defaultdict(list) for t in IOU_THRESHOLDS}
    conf_matrix  = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1))

    t0 = time.time()

    for idx, item in enumerate(test_data):
        gts     = load_gt(item["label"])
        results = model.predict(source=item["image"], conf=CONF,
                                verbose=False, device=DEVICE, imgsz=IMGSZ)
        preds = []
        for b in results[0].boxes:
            cid  = int(b.cls[0])
            conf = float(b.conf[0])
            box  = b.xywhn[0].tolist()
            if is_baseline:
                if cid in COCO_TO_RF:
                    preds.append({"class": COCO_TO_RF[cid], "box": box, "conf": conf})
            else:
                if cid in RF_NAMES:
                    preds.append({"class": cid, "box": box, "conf": conf})

        matches, unmatched_p, unmatched_g = greedy_match_fixed(preds, gts, 0.5)

        for pi, gi, iou in matches:
            c = preds[pi]["class"]
            class_stats[c]["TP"] += 1
            period_stats[item["period"]][c]["TP"] += 1
            camera_stats[item["camera"]]["TP"] += 1
            per_class_confs[c].append((preds[pi]["conf"], True))
            conf_matrix[c][c] += 1

        for pi in unmatched_p:
            c = preds[pi]["class"]
            class_stats[c]["FP"] += 1
            period_stats[item["period"]][c]["FP"] += 1
            camera_stats[item["camera"]]["FP"] += 1
            per_class_confs[c].append((preds[pi]["conf"], False))
            conf_matrix[c][NUM_CLASSES] += 1

        for gi in unmatched_g:
            c = gts[gi]["class"]
            class_stats[c]["FN"] += 1
            period_stats[item["period"]][c]["FN"] += 1
            camera_stats[item["camera"]]["FN"] += 1
            conf_matrix[NUM_CLASSES][c] += 1

        for g in gts:
            camera_stats[item["camera"]]["GT"] += 1

        for pi in unmatched_p:
            pred_c = preds[pi]["class"]
            best_iou, best_gt_c = 0, None
            for gi in unmatched_g:
                iou = compute_iou(preds[pi]["box"], gts[gi]["box"])
                if iou > best_iou and iou >= 0.3:
                    best_iou = iou; best_gt_c = gts[gi]["class"]
            if best_gt_c is not None and best_gt_c != pred_c:
                conf_matrix[pred_c][best_gt_c] += 1
                conf_matrix[pred_c][NUM_CLASSES] -= 1

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

    per_class, all_aps_50, per_class_ap50_95 = [], [], []

    for cid in sorted(RF_NAMES.keys()):
        s = class_stats[cid]
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        gt = tp + fn
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        ap50 = compute_ap_11point(per_class_confs[cid], gt)
        all_aps_50.append(ap50)
        aps_at_thresholds = []
        for iou_thr in IOU_THRESHOLDS:
            k = round(iou_thr, 2)
            aps_at_thresholds.append(compute_ap_11point(per_class_confs_strict[k][cid], gt))
        ap50_95 = np.mean(aps_at_thresholds)
        per_class_ap50_95.append(ap50_95)
        per_class.append({"cid": cid, "name": RF_NAMES[cid], "GT": gt,
                           "TP": tp, "FP": fp, "FN": fn,
                           "P": p, "R": r, "F1": f1,
                           "AP50": ap50, "AP50_95": ap50_95})

    mAP50    = np.mean(all_aps_50)
    mAP50_95 = np.mean(per_class_ap50_95)

    ttp = sum(class_stats[c]["TP"] for c in RF_NAMES)
    tfp = sum(class_stats[c]["FP"] for c in RF_NAMES)
    tfn = sum(class_stats[c]["FN"] for c in RF_NAMES)
    op  = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0
    orr = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0
    of1 = 2 * op * orr / (op + orr) if (op + orr) > 0 else 0

    day_night = {}
    for period in ["day", "night"]:
        ps  = period_stats[period]
        ptp = sum(ps[c]["TP"] for c in RF_NAMES)
        pfp = sum(ps[c]["FP"] for c in RF_NAMES)
        pfn = sum(ps[c]["FN"] for c in RF_NAMES)
        pp  = ptp / (ptp + pfp) if (ptp + pfp) > 0 else 0
        pr  = ptp / (ptp + pfn) if (ptp + pfn) > 0 else 0
        pf  = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0
        ni  = sum(1 for d in test_data if d["period"] == period)
        day_night[period] = {"imgs": ni, "P": pp, "R": pr, "F1": pf}

    per_camera = []
    for cam in sorted(camera_stats.keys(),
                      key=lambda c: camera_stats[c]["GT"], reverse=True):
        s  = camera_stats[cam]
        cp = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0
        cr = s["TP"] / s["GT"] if s["GT"] > 0 else 0
        per_camera.append({"camera": cam, "GT": s["GT"], "P": cp, "R": cr})

    # Print results
    print(f"\n  Per-class results:")
    header = f"  {'Class':18s} {'GT':>5s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AP@0.5':>8s} {'AP@.5:.95':>10s}"
    print(header); print(f"  {'-'*66}")
    for pc in per_class:
        print(f"  {pc['name']:18s} {pc['GT']:5d} {pc['P']:6.1%} {pc['R']:6.1%} {pc['F1']:6.1%} {pc['AP50']:7.1%} {pc['AP50_95']:9.1%}")
    tgt = ttp + tfn
    print(f"\n  {'OVERALL':18s} {tgt:5d} {op:6.1%} {orr:6.1%} {of1:6.1%} {mAP50:7.1%} {mAP50_95:9.1%}")
    print(f"\n  mAP@0.5      = {mAP50:.4f} ({mAP50:.1%})")
    print(f"  mAP@0.5:0.95 = {mAP50_95:.4f} ({mAP50_95:.1%})")
    print(f"\n  Day vs Night:")
    for period in ["day", "night"]:
        s = day_night[period]
        print(f"    {period:6s}: {s['imgs']:3d} imgs  P={s['P']:.1%}  R={s['R']:.1%}  F1={s['F1']:.1%}")

    return {
        "model": model_name,
        "overall": {"P": op, "R": orr, "F1": of1, "mAP50": mAP50, "mAP50_95": mAP50_95},
        "per_class": per_class,
        "day_night": day_night,
        "per_camera": per_camera,
        "confusion_matrix": conf_matrix,
    }


# ============================================================
# SAVE RESULTS TO data/results_v2/
# ============================================================
def save_results(rq1, rq2):
    # Overall
    with open(RESULTS_DIR / "overall_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "yolov8s_coco", "yolov8s_finetuned", "delta"])
        for key in ["P", "R", "F1", "mAP50", "mAP50_95"]:
            bv = rq1["overall"][key]; fv = rq2["overall"][key]
            w.writerow([key, round(bv, 4), round(fv, 4), round(fv - bv, 4)])

    # Per-class
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

    # Day/Night
    with open(RESULTS_DIR / "day_night_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "bl_imgs", "bl_P", "bl_R", "bl_F1",
                    "ft_imgs", "ft_P", "ft_R", "ft_F1"])
        for period in ["day", "night"]:
            b = rq1["day_night"][period]; ft = rq2["day_night"][period]
            w.writerow([period, b["imgs"], round(b["P"],4), round(b["R"],4), round(b["F1"],4),
                        ft["imgs"], round(ft["P"],4), round(ft["R"],4), round(ft["F1"],4)])

    # Per-camera
    with open(RESULTS_DIR / "per_camera_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "GT", "bl_P", "bl_R", "ft_P", "ft_R"])
        bl_cams = {c["camera"]: c for c in rq1["per_camera"]}
        for fc in rq2["per_camera"]:
            cam = fc["camera"]; bc = bl_cams.get(cam, {"P": 0, "R": 0})
            w.writerow([cam, fc["GT"],
                        round(bc["P"],4), round(bc["R"],4),
                        round(fc["P"],4), round(fc["R"],4)])

    labels = [RF_NAMES[i] for i in sorted(RF_NAMES.keys())] + ["Background"]
    np.savetxt(RESULTS_DIR / "confusion_matrix_finetuned.csv",
               rq2["confusion_matrix"], delimiter=",", fmt="%.0f",
               header=",".join(labels), comments="")

    print(f"\n  All results saved to {RESULTS_DIR}/")
    print(f"  Original v1 results in data/results/ are UNTOUCHED")


# ============================================================
# LOAD TEST DATA
# ============================================================
print("\n" + "="*70)
print("  RQ1 + RQ2 EVALUATION — VERSION 2")
print("  Baseline: YOLOv8s COCO (architecture-controlled comparison)")
print("  Fine-tuned: YOLOv8s Newcastle v6")
print("  Test set: SAME 344 images as v1 — fair comparison guaranteed")
print("="*70)

test_data = []
img_dir   = TEST_DIR / "images"
lbl_dir   = TEST_DIR / "labels"

for img_path in sorted(img_dir.glob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    parts  = img_path.stem.split("__")
    camera = parts[0] if len(parts) >= 2 else img_path.stem.split("_")[0]
    hour   = parse_hour(img_path.name)
    period = "day" if hour is not None and 7 <= hour < 18 else "night"
    test_data.append({
        "image": str(img_path), "label": str(lbl_dir / f"{img_path.stem}.txt"),
        "filename": img_path.name, "camera": camera, "hour": hour, "period": period
    })

gt_counts = defaultdict(int)
for d in test_data:
    for gt in load_gt(d["label"]):
        gt_counts[gt["class"]] += 1

day_n   = sum(1 for d in test_data if d["period"] == "day")
night_n = sum(1 for d in test_data if d["period"] == "night")
n_cams  = len(set(d["camera"] for d in test_data))
total_gt = sum(gt_counts.values())

print(f"\n  Test images : {len(test_data)} ({day_n} day, {night_n} night)")
print(f"  Cameras     : {n_cams}")
print(f"  Ground truth: {total_gt} objects across {len(gt_counts)} classes")


# ============================================================
# RUN EVALUATIONS
# ============================================================
print(f"\n  Loading baseline: {BASELINE_MODEL_PATH}")
baseline_model = YOLO(BASELINE_MODEL_PATH)
rq1 = evaluate_model(baseline_model, test_data,
                     "Baseline YOLOv8s (COCO)", is_baseline=True)

print(f"\n  Loading fine-tuned: {FINETUNED_MODEL_PATH}")
finetuned_model = YOLO(FINETUNED_MODEL_PATH)
rq2 = evaluate_model(finetuned_model, test_data,
                     "Fine-tuned YOLOv8s (Newcastle v6, 789 imgs)", is_baseline=False)


# ============================================================
# COMPARISON SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"  COMPARISON: YOLOv8s COCO vs YOLOv8s Fine-tuned")
print(f"  (Both same architecture — delta shows domain adaptation only)")
print(f"{'='*70}")
header = f"  {'Metric':<18} {'YOLOv8s COCO':>14} {'YOLOv8s FT':>12} {'Delta':>10}"
print(header); print(f"  {'-'*56}")
for key, label in [("P","Precision"),("R","Recall"),("F1","F1 Score"),
                   ("mAP50","mAP@0.5"),("mAP50_95","mAP@0.5:0.95")]:
    bv = rq1["overall"][key]; fv = rq2["overall"][key]; d = fv - bv
    sign = "+" if d > 0 else ""
    print(f"  {label:<18} {bv:>13.1%} {fv:>11.1%} {sign}{d:>8.1%}")

print(f"\n  Per-class AP@0.5:")
header2 = f"  {'Class':18s} {'v8s COCO':>9s} {'v8s FT':>9s} {'Delta':>9s}"
print(header2); print(f"  {'-'*48}")
for bp, fp in zip(rq1["per_class"], rq2["per_class"]):
    d = fp["AP50"] - bp["AP50"]
    sign = "+" if d > 0 else ""
    print(f"  {bp['name']:18s} {bp['AP50']:>8.1%} {fp['AP50']:>8.1%} {sign}{d:>8.1%}")

print(f"\n{'='*70}")
print(f"  4-WAY TABLE (add to dissertation Section IV-B):")
print(f"{'='*70}")
print(f"  {'Model':<35} {'mAP@0.5':>8} {'mAP@5:95':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*80}")
print(f"  {'YOLOv8n COCO (from v1 results)':35}   12.9%      9.5%      70.7%   17.9%   28.5%")
print(f"  {'YOLOv8s COCO (this run)':35} {rq1['overall']['mAP50']:>7.1%} {rq1['overall']['mAP50_95']:>9.1%} {rq1['overall']['P']:>9.1%} {rq1['overall']['R']:>7.1%} {rq1['overall']['F1']:>7.1%}")
print(f"  {'YOLOv8s fine-tuned v6 (this run)':35} {rq2['overall']['mAP50']:>7.1%} {rq2['overall']['mAP50_95']:>9.1%} {rq2['overall']['P']:>9.1%} {rq2['overall']['R']:>7.1%} {rq2['overall']['F1']:>7.1%}")
print(f"\n  Paste these numbers into the paper table.")
print(f"  YOLOv8n fine-tuned row: run rq2_finetune.py --model yolov8n.pt if needed.")


# ============================================================
# SAVE ALL RESULTS
# ============================================================
save_results(rq1, rq2)

print(f"\n{'='*70}")
print(f"  DONE — v2 complete")
print(f"  Results in: data/results_v2/")
print(f"  Original:   data/results/  (untouched)")
print(f"{'='*70}")
