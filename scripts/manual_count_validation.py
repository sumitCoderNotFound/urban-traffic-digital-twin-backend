"""
RQ3 Manual Count Validation
============================
Validates automated vehicle counts from the deployed model against
human-generated ground truth counts for a stratified random sample
of RQ3 frames.

WORKFLOW
--------
Step 1 — Generate sample (run once):
    python scripts/manual_count_validation.py --mode sample \
        --csv data/results/auto_label_detections.csv \
        --n 80 --output data/manual_validation/

    This creates:
        data/manual_validation/sample_frames.csv   — frames to count
        data/manual_validation/                    — folder of sample images

Step 2 — Manual counting (researcher task):
    Open data/manual_validation/sample_frames.csv in Excel / LibreOffice.
    For each row, open the linked image_path and count vehicles VISIBLE in frame.
    Fill in column  manual_vehicle_count  with your count.
    Save the file.

Step 3 — Compute statistics (run once Step 2 is done):
    python scripts/manual_count_validation.py --mode validate \
        --manual data/manual_validation/sample_frames.csv \
        --output data/manual_validation/

    This produces:
        data/manual_validation/validation_results.csv
        data/manual_validation/validation_summary.txt
        data/manual_validation/fig_manual_validation.png

Author: Sumit Malviya (W24041293)
Supervisor: Dr. Jason Moore
Module: KF7029 — Northumbria University
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RQ3 Manual Count Validation")
parser.add_argument("--mode", choices=["sample", "validate"], required=True,
                    help="'sample': draw frames for manual counting | "
                         "'validate': compute statistics after manual counts filled in")
parser.add_argument("--csv", default="data/results/auto_label_detections.csv",
                    help="[sample mode] Path to RQ3 auto_label detections CSV")
parser.add_argument("--n", type=int, default=80,
                    help="[sample mode] Number of frames to sample (default 80)")
parser.add_argument("--manual", default="data/manual_validation/sample_frames.csv",
                    help="[validate mode] Path to completed manual count CSV")
parser.add_argument("--output", default="data/manual_validation/",
                    help="Output directory for results")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default 42)")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SAMPLE
# ═══════════════════════════════════════════════════════════════════════════════
if args.mode == "sample":

    CSV_PATH = Path(args.csv)
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        print("        Run the RQ3 bulk analysis first to generate this file.")
        sys.exit(1)

    # ── Load detections ───────────────────────────────────────────────────────
    records = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "image":      row.get("image_path", row.get("image", "")),
                    "camera":     row.get("camera", "unknown"),
                    "hour":       int(row.get("hour", -1)),
                    "period":     row.get("period", "day"),
                    "vehicles":   int(float(row.get("vehicles", 0))),
                    "confidence": float(row.get("avg_confidence", row.get("confidence", 0))),
                })
            except (ValueError, KeyError):
                continue

    print(f"Loaded {len(records):,} detection records")
    if not records:
        print("[ERROR] No usable records found in CSV.")
        sys.exit(1)

    # ── Stratified sampling ───────────────────────────────────────────────────
    # Strata: low (0 veh), medium (1-4 veh), high (5+ veh) x day/night
    # Ensures sample covers the full distribution, not just busy frames.
    STRATA_LABELS = ["low_day", "low_night", "medium_day", "medium_night",
                     "high_day", "high_night"]

    def get_stratum(rec):
        v = rec["vehicles"]
        p = rec["period"]
        if v == 0:
            tier = "low"
        elif v <= 4:
            tier = "medium"
        else:
            tier = "high"
        return f"{tier}_{p}"

    strata = defaultdict(list)
    for rec in records:
        strata[get_stratum(rec)].append(rec)

    print("\nStratum sizes:")
    for s in STRATA_LABELS:
        print(f"  {s:15s}: {len(strata[s]):5,}")

    # Allocate sample proportionally but with minimum 5 per stratum
    total = len(records)
    target_n = args.n
    alloc = {}
    for s in STRATA_LABELS:
        prop = len(strata[s]) / total
        alloc[s] = max(5, round(prop * target_n))

    # Trim to target if over
    while sum(alloc.values()) > target_n:
        biggest = max(alloc, key=alloc.get)
        alloc[biggest] -= 1

    print(f"\nTarget sample: {target_n} frames")
    print("Allocation:")
    for s in STRATA_LABELS:
        print(f"  {s:15s}: {alloc[s]:3d} frames")

    # Draw sample
    random.seed(args.seed)
    sample = []
    for s in STRATA_LABELS:
        pool = strata[s]
        n = min(alloc[s], len(pool))
        drawn = random.sample(pool, n)
        for rec in drawn:
            rec["stratum"] = s
        sample.extend(drawn)

    random.shuffle(sample)

    # ── Write CSV for manual counting ─────────────────────────────────────────
    out_csv = OUTPUT_DIR / "sample_frames.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id", "image_path", "camera", "hour", "period",
            "stratum", "model_vehicle_count", "model_confidence",
            "manual_vehicle_count",   # ← RESEARCHER FILLS THIS IN
            "notes"                    # ← OPTIONAL NOTES
        ])
        for i, rec in enumerate(sample, 1):
            writer.writerow([
                i,
                rec["image"],
                rec["camera"],
                rec["hour"],
                rec["period"],
                rec["stratum"],
                rec["vehicles"],
                round(rec["confidence"], 3),
                "",    # manual_vehicle_count — TO BE FILLED
                ""     # notes
            ])

    # ── Copy sample images if they exist ─────────────────────────────────────
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)
    copied = 0
    for i, rec in enumerate(sample, 1):
        src = Path(rec["image"])
        if src.exists():
            ext = src.suffix
            dst = img_dir / f"{i:03d}_{src.name}"
            shutil.copy2(src, dst)
            copied += 1

    print(f"\nSample CSV written: {out_csv}")
    print(f"Images copied:      {copied}/{len(sample)}")
    print("\n" + "="*60)
    print("NEXT STEP:")
    print(f"  1. Open {out_csv} in a spreadsheet")
    print("  2. For each row open the image at image_path")
    print("  3. Count ALL visible vehicles in the frame")
    print("  4. Enter the count in column 'manual_vehicle_count'")
    print("  5. Save and run:")
    print(f"     python scripts/manual_count_validation.py --mode validate "
          f"--manual {out_csv} --output {OUTPUT_DIR}")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════
else:

    MANUAL_CSV = Path(args.manual)
    if not MANUAL_CSV.exists():
        print(f"[ERROR] Manual count CSV not found: {MANUAL_CSV}")
        sys.exit(1)

    # ── Load completed counts ─────────────────────────────────────────────────
    rows = []
    with open(MANUAL_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manual_raw = row.get("manual_vehicle_count", "").strip()
            if not manual_raw:
                continue   # skip rows not yet filled
            try:
                rows.append({
                    "frame_id":  int(row["frame_id"]),
                    "camera":    row["camera"],
                    "hour":      int(row["hour"]),
                    "period":    row["period"],
                    "stratum":   row["stratum"],
                    "model":     int(float(row["model_vehicle_count"])),
                    "manual":    int(float(manual_raw)),
                    "conf":      float(row.get("model_confidence", 0)),
                })
            except (ValueError, KeyError) as e:
                print(f"  [WARN] Skipping row {row.get('frame_id','?')}: {e}")

    n = len(rows)
    if n < 2:
        print(f"[ERROR] Only {n} completed row(s). Fill in at least 2 rows.")
        sys.exit(1)

    model_counts  = np.array([r["model"]  for r in rows])
    manual_counts = np.array([r["manual"] for r in rows])

    # ── Core statistics ───────────────────────────────────────────────────────
    errors    = model_counts - manual_counts
    mae       = float(np.mean(np.abs(errors)))
    rmse      = float(np.sqrt(np.mean(errors ** 2)))
    mpe       = float(np.mean(errors))          # mean prediction error (bias)
    mape_vals = []
    for m, a in zip(model_counts, manual_counts):
        if a > 0:
            mape_vals.append(abs(m - a) / a)
    mape = float(np.mean(mape_vals) * 100) if mape_vals else float("nan")

    # Pearson r
    if np.std(model_counts) > 0 and np.std(manual_counts) > 0:
        r = float(np.corrcoef(model_counts, manual_counts)[0, 1])
    else:
        r = float("nan")

    # R²
    ss_res = float(np.sum((manual_counts - model_counts) ** 2))
    ss_tot = float(np.sum((manual_counts - np.mean(manual_counts)) ** 2))
    r_sq   = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Agreement within ±1 and ±2 vehicles
    within_1 = float(np.mean(np.abs(errors) <= 1) * 100)
    within_2 = float(np.mean(np.abs(errors) <= 2) * 100)

    # ── Per-stratum breakdown ─────────────────────────────────────────────────
    strata_stats = defaultdict(lambda: {"model": [], "manual": []})
    for r_row in rows:
        strata_stats[r_row["stratum"]]["model"].append(r_row["model"])
        strata_stats[r_row["stratum"]]["manual"].append(r_row["manual"])

    stratum_results = {}
    for s, d in strata_stats.items():
        mv = np.array(d["model"]); man = np.array(d["manual"])
        err = mv - man
        stratum_results[s] = {
            "n":    len(mv),
            "mae":  float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
        }

    # Day vs night MAE
    day_rows   = [r_row for r_row in rows if r_row["period"] == "day"]
    night_rows = [r_row for r_row in rows if r_row["period"] == "night"]
    day_mae   = float(np.mean(np.abs(np.array([r_row["model"] for r_row in day_rows]) -
                                     np.array([r_row["manual"] for r_row in day_rows])))) \
                if day_rows else float("nan")
    night_mae = float(np.mean(np.abs(np.array([r_row["model"] for r_row in night_rows]) -
                                     np.array([r_row["manual"] for r_row in night_rows])))) \
                if night_rows else float("nan")

    # ── Print summary ─────────────────────────────────────────────────────────
    summary_lines = [
        "="*62,
        "  RQ3 MANUAL COUNT VALIDATION RESULTS",
        "="*62,
        f"  Frames validated : {n}",
        f"  Model count range: {int(model_counts.min())} – {int(model_counts.max())}",
        f"  Manual count range: {int(manual_counts.min())} – {int(manual_counts.max())}",
        "",
        "  ACCURACY METRICS",
        f"  MAE              : {mae:.3f} vehicles/frame",
        f"  RMSE             : {rmse:.3f} vehicles/frame",
        f"  Mean bias (MPE)  : {mpe:+.3f} vehicles/frame  "
        f"({'over' if mpe > 0 else 'under'}-counting)",
        f"  MAPE             : {mape:.1f}%",
        f"  Pearson r        : {r:.4f}",
        f"  R²               : {r_sq:.4f}",
        f"  Within ±1 veh    : {within_1:.1f}%",
        f"  Within ±2 veh    : {within_2:.1f}%",
        "",
        "  DAY / NIGHT",
        f"  Day MAE          : {day_mae:.3f}  (n={len(day_rows)})",
        f"  Night MAE        : {night_mae:.3f}  (n={len(night_rows)})",
        "",
        "  PER-STRATUM MAE",
    ]
    for s in sorted(stratum_results):
        st = stratum_results[s]
        summary_lines.append(
            f"  {s:18s}: MAE={st['mae']:.3f}  bias={st['bias']:+.3f}  (n={st['n']})"
        )
    summary_lines.append("="*62)

    for line in summary_lines:
        print(line)

    # ── Save summary text ─────────────────────────────────────────────────────
    summary_txt = OUTPUT_DIR / "validation_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary saved: {summary_txt}")

    # ── Save per-row results CSV ───────────────────────────────────────────────
    results_csv = OUTPUT_DIR / "validation_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id", "camera", "hour", "period", "stratum",
            "model_count", "manual_count", "error", "abs_error"
        ])
        for r_row in rows:
            err = r_row["model"] - r_row["manual"]
            writer.writerow([
                r_row["frame_id"], r_row["camera"], r_row["hour"],
                r_row["period"], r_row["stratum"],
                r_row["model"], r_row["manual"], err, abs(err)
            ])
    print(f"Per-row CSV saved: {results_csv}")

    # ── Figure: scatter + error distribution ─────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle(
            f"RQ3 Manual Count Validation  (n={n})",
            fontsize=12, fontweight="bold", y=1.01
        )

        BLUE   = "#185FA5"
        ORANGE = "#BA7517"
        GREEN  = "#2E7D32"

        # Panel A — scatter model vs manual
        ax = axes[0]
        mx = max(model_counts.max(), manual_counts.max()) + 1
        ax.scatter(manual_counts, model_counts, alpha=0.6, color=BLUE,
                   edgecolors="white", linewidths=0.5, s=50, zorder=3)
        ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.4, label="Perfect agreement")

        # Regression line
        if len(rows) >= 3:
            slope, intercept, *_ = scipy_stats.linregress(manual_counts, model_counts)
            xs = np.linspace(0, mx, 50)
            ax.plot(xs, slope * xs + intercept, color=ORANGE, lw=1.5,
                    label=f"Fit  r={r:.3f}")

        ax.set_xlabel("Manual count (ground truth)")
        ax.set_ylabel("Model count")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.legend(fontsize=8)
        ax.set_title(f"Agreement  (MAE={mae:.2f})", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel B — error histogram
        ax = axes[1]
        bins = range(int(errors.min()) - 1, int(errors.max()) + 2)
        ax.hist(errors, bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(mpe, color=ORANGE, lw=1.5, ls="-", label=f"Bias={mpe:+.2f}")
        ax.set_xlabel("Error (model − manual)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Error distribution  (RMSE={rmse:.2f})", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel C — per-stratum MAE
        ax = axes[2]
        s_names = sorted(stratum_results)
        s_maes  = [stratum_results[s]["mae"]  for s in s_names]
        s_ns    = [stratum_results[s]["n"]    for s in s_names]
        colors  = [BLUE if "day" in s else ORANGE for s in s_names]
        bars = ax.barh(s_names, s_maes, color=colors, height=0.55)
        for bar, n_val in zip(bars, s_ns):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"n={n_val}", va="center", fontsize=7)
        ax.set_xlabel("MAE (vehicles/frame)")
        ax.set_title("MAE by traffic stratum", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend for day/night colours
        from matplotlib.patches import Patch
        legend_els = [Patch(color=BLUE, label="Day"),
                      Patch(color=ORANGE, label="Night")]
        ax.legend(handles=legend_els, fontsize=8)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "fig_manual_validation.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Validation figure: {fig_path}")

    except ImportError:
        print("[WARN] matplotlib/scipy not available — figure not generated")

    # ── Save JSON summary for paper ───────────────────────────────────────────
    json_summary = {
        "n_validated": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_bias": round(mpe, 4),
        "mape_pct": round(mape, 2),
        "pearson_r": round(r, 4),
        "r_squared": round(r_sq, 4),
        "within_1_pct": round(within_1, 1),
        "within_2_pct": round(within_2, 1),
        "day_mae": round(day_mae, 4) if not math.isnan(day_mae) else None,
        "night_mae": round(night_mae, 4) if not math.isnan(night_mae) else None,
        "per_stratum": stratum_results,
    }
    json_path = OUTPUT_DIR / "validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"JSON summary:      {json_path}")

    print("\n" + "="*62)
    print("  KEY NUMBERS FOR PAPER (Section IV.C / V):")
    print(f"  MAE = {mae:.3f} vehicles/frame,  r = {r:.3f},  R² = {r_sq:.3f}")
    print(f"  {within_1:.0f}% of predictions within ±1 vehicle of ground truth")
    print("="*62)
