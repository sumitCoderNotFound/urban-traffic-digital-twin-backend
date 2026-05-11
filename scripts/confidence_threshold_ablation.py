"""
Confidence Threshold Ablation
==============================
Evaluates the fine-tuned YOLOv8s model at 4 confidence thresholds
to find the optimal operating point for Newcastle CCTV imagery.

Results saved to: data/results_ablation/confidence_threshold_ablation.csv

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
MODEL_PATH  = "runs/detect/newcastle_v6_improved/weights/best.pt"
DATASET_DIR = Path("data/Newcastle-Traffic-Detection.v6i.yolov8")
TEST_DIR    = DATASET_DIR / "test"
RESULTS_DIR = Path("data/results_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.15, 0.25, 0.35, 0.50]
DEVICE     = "cpu"
IMGSZ      = 960

RF_NAMES   = {0:"Motorcycle", 1:"Roadwork", 2:"Traffic light", 3:"bicycle",
              4:"bus", 5:"car", 6:"person", 7:"truck"}
NUM_CLASSES = len(RF_NAMES)


# ============================================================
# HELPERS
# ============================================================
def compute_iou(b1, b2):
    x1 = max(b1[0]-b1[2]/2, b2[0]-b2[2]/2)
    y1 = max(b1[1]-b1[3]/2, b2[1]-b2[3]/2)
    x2 = min(b1[0]+b1[2]/2, b2[0]+b2[2]/2)
    y2 = min(b1[1]+b1[3]/2, b2[1]+b2[3]/2)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0


def greedy_match(preds, gts, iou_thr=0.5):
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
    return matches, [i for i in range(len(preds)) if i not in matched_p], \
                    [i for i in range(len(gts))  if i not in matched_g]


def compute_ap(confs_tps, total_gt):
    if not confs_tps or total_gt == 0: return 0.0
    sorted_dets = sorted(confs_tps, key=lambda x: x[0], reverse=True)
    tp = fp = 0
    ps, rs = [], []
    for conf, is_tp in sorted_dets:
        if is_tp: tp += 1
        else:     fp += 1
        ps.append(tp / (tp + fp))
        rs.append(tp / total_gt)
    return sum(max([p for p,r in zip(ps,rs) if r >= t], default=0)
               for t in np.arange(0, 1.1, 0.1)) / 11.0


def load_gt(path):
    boxes = []
    if not os.path.exists(path): return boxes
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append({"class": int(parts[0]),
                          "box": [float(p) for p in parts[1:5]]})
    return boxes


def parse_hour(filename):
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        h = int(m.group(2)[:2])
        return h if 0 <= h <= 23 else None
    return None


# ============================================================
# LOAD TEST DATA
# ============================================================
test_data = []
for img_path in sorted((TEST_DIR / "images").glob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
    parts  = img_path.stem.split("__")
    camera = parts[0] if len(parts) >= 2 else img_path.stem.split("_")[0]
    hour   = parse_hour(img_path.name)
    period = "day" if hour is not None and 7 <= hour < 18 else "night"
    test_data.append({
        "image":  str(img_path),
        "label":  str(TEST_DIR / "labels" / f"{img_path.stem}.txt"),
        "camera": camera, "period": period
    })

# Preload all predictions at conf=0.10 (lowest threshold)
# Then filter per threshold — avoids running YOLO 4 times
print(f"\n{'='*60}")
print(f"  CONFIDENCE THRESHOLD ABLATION")
print(f"  Model: {MODEL_PATH}")
print(f"  Thresholds: {THRESHOLDS}")
print(f"  Test images: {len(test_data)}")
print(f"{'='*60}\n")

print("Loading model and running inference at conf=0.10 (collect all detections)...")
model = YOLO(MODEL_PATH)

# Run at very low threshold to capture everything, filter later
all_preds = []  # list of lists, one per image
t0 = time.time()
for idx, item in enumerate(test_data):
    results = model.predict(source=item["image"], conf=0.10,
                            verbose=False, device=DEVICE, imgsz=IMGSZ)
    preds = []
    for b in results[0].boxes:
        cid  = int(b.cls[0])
        if cid in RF_NAMES:
            preds.append({"class": cid,
                          "box":   b.xywhn[0].tolist(),
                          "conf":  float(b.conf[0])})
    all_preds.append(preds)
    if (idx + 1) % 100 == 0:
        print(f"  [{idx+1}/{len(test_data)}]")

print(f"  Inference done in {time.time()-t0:.0f}s\n")

# ============================================================
# EVALUATE AT EACH THRESHOLD
# ============================================================
results_summary = []

for conf_thr in THRESHOLDS:
    print(f"  --- Evaluating conf={conf_thr} ---")

    class_stats  = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    per_class_confs = defaultdict(list)
    gt_counts    = defaultdict(int)

    for idx, item in enumerate(test_data):
        gts   = load_gt(item["label"])
        preds = [p for p in all_preds[idx] if p["conf"] >= conf_thr]

        for g in gts:
            gt_counts[g["class"]] += 1

        matches, unmatched_p, unmatched_g = greedy_match(preds, gts, 0.5)

        for pi, gi, _ in matches:
            c = preds[pi]["class"]
            class_stats[c]["TP"] += 1
            per_class_confs[c].append((preds[pi]["conf"], True))

        for pi in unmatched_p:
            c = preds[pi]["class"]
            class_stats[c]["FP"] += 1
            per_class_confs[c].append((preds[pi]["conf"], False))

        for gi in unmatched_g:
            c = gts[gi]["class"]
            class_stats[c]["FN"] += 1

    # Overall metrics
    ttp = sum(class_stats[c]["TP"] for c in RF_NAMES)
    tfp = sum(class_stats[c]["FP"] for c in RF_NAMES)
    tfn = sum(class_stats[c]["FN"] for c in RF_NAMES)
    op  = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0
    orr = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0
    of1 = 2 * op * orr / (op + orr) if (op + orr) > 0 else 0

    # mAP@0.5
    aps = []
    for cid in sorted(RF_NAMES.keys()):
        gt = gt_counts[cid]
        ap = compute_ap(per_class_confs[cid], gt)
        aps.append(ap)
    mAP50 = np.mean(aps)

    results_summary.append({
        "conf": conf_thr, "P": op, "R": orr, "F1": of1, "mAP50": mAP50
    })

    print(f"    P={op:.1%}  R={orr:.1%}  F1={of1:.1%}  mAP@0.5={mAP50:.1%}")


# ============================================================
# PRINT FINAL TABLE
# ============================================================
print(f"\n{'='*60}")
print(f"  CONFIDENCE THRESHOLD ABLATION — RESULTS")
print(f"{'='*60}")
print(f"  {'Conf':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'mAP@0.5':>9}")
print(f"  {'-'*50}")

best_f1   = max(results_summary, key=lambda x: x["F1"])
best_map  = max(results_summary, key=lambda x: x["mAP50"])

for r in results_summary:
    marker = " <-- best F1" if r["conf"] == best_f1["conf"] else ""
    marker = " <-- best mAP" if r["conf"] == best_map["conf"] and r["conf"] != best_f1["conf"] else marker
    print(f"  {r['conf']:>6.2f}  {r['P']:>9.1%}  {r['R']:>7.1%}  {r['F1']:>7.1%}  {r['mAP50']:>8.1%}{marker}")

print(f"\n  Best F1   at conf={best_f1['conf']}: F1={best_f1['F1']:.1%}")
print(f"  Best mAP  at conf={best_map['conf']}: mAP={best_map['mAP50']:.1%}")
print(f"\n  Recommendation: use conf={best_f1['conf']} for deployment")
print(f"  (highest F1 = best precision/recall balance)")


# ============================================================
# SAVE CSV
# ============================================================
csv_path = RESULTS_DIR / "confidence_threshold_ablation.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["conf_threshold", "precision", "recall", "f1", "mAP50"])
    for r in results_summary:
        w.writerow([r["conf"], round(r["P"],4), round(r["R"],4),
                    round(r["F1"],4), round(r["mAP50"],4)])

print(f"\n  Saved: {csv_path}")
print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
