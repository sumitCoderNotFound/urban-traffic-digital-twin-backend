"""
Compute Camera Percentiles
===========================
Analyses all detection records and computes empirical percentile
thresholds for the traffic calculator.

Run this ONCE before deployment, or periodically to update baselines.

Usage:
    # From auto_label CSV (from the 20K image analysis):
    python scripts/compute_percentiles.py --source csv --csv data/results/auto_label_detections.csv

    # From database (if scheduler has been running):
    python scripts/compute_percentiles.py --source db

Author: Sumit Malviya (W24041293)
"""

import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description="Compute camera percentile thresholds")
parser.add_argument("--source", choices=["db", "csv"], default="csv",
                    help="Data source: 'db' for database, 'csv' for auto_label CSV")
parser.add_argument("--csv", default="data/results/auto_label_detections.csv",
                    help="Path to auto_label_detections.csv")
parser.add_argument("--output", default="data/results/camera_percentiles.json",
                    help="Output JSON path")
args = parser.parse_args()


def load_from_csv(csv_path):
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "camera_id": row.get("camera", "unknown"),
                "vehicles": int(row.get("vehicles", 0)),
                "pedestrians": int(row.get("pedestrians", 0)),
                "cyclists": int(row.get("cyclists", 0)),
            })
    return records


def load_from_db():
    import sqlite3
    db_path = "urban_twin.db"
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT camera_id, vehicles, pedestrians, cyclists FROM detections"
    )
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records


def compute_camera_percentiles(detections, output_path):
    camera_data = defaultdict(lambda: {"vehicles": [], "pedestrians": []})

    for det in detections:
        cam = det.get("camera_id", det.get("camera", "unknown"))
        camera_data[cam]["vehicles"].append(det.get("vehicles", 0))
        camera_data[cam]["pedestrians"].append(det.get("pedestrians", 0))

    results = {}
    all_vehicles = []
    all_pedestrians = []

    for cam_id, data in camera_data.items():
        v = np.array(data["vehicles"])
        p = np.array(data["pedestrians"])
        all_vehicles.extend(data["vehicles"])
        all_pedestrians.extend(data["pedestrians"])

        results[cam_id] = {
            "n_observations": len(v),
            "p25_vehicles": float(np.percentile(v, 25)),
            "p75_vehicles": float(np.percentile(v, 75)),
            "p95_vehicles": float(np.percentile(v, 95)),
            "mean_vehicles": float(np.mean(v)),
            "p25_pedestrians": float(np.percentile(p, 25)),
            "p75_pedestrians": float(np.percentile(p, 75)),
            "p95_pedestrians": float(np.percentile(p, 95)),
            "mean_pedestrians": float(np.mean(p)),
        }

    if all_vehicles:
        av = np.array(all_vehicles)
        ap = np.array(all_pedestrians)
        results["__global__"] = {
            "n_observations": len(av),
            "p25_vehicles": float(np.percentile(av, 25)),
            "p75_vehicles": float(np.percentile(av, 75)),
            "p95_vehicles": float(np.percentile(av, 95)),
            "mean_vehicles": float(np.mean(av)),
            "p25_pedestrians": float(np.percentile(ap, 25)),
            "p75_pedestrians": float(np.percentile(ap, 75)),
            "p95_pedestrians": float(np.percentile(ap, 95)),
            "mean_pedestrians": float(np.mean(ap)),
        }

    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Camera percentiles computed for {len(results) - 1} cameras")
    print(f"Saved to {save_path}")
    return results


# ============================================================
# MAIN
# ============================================================
print(f"Loading detections from {args.source}...")

if args.source == "csv":
    if not Path(args.csv).exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)
    detections = load_from_csv(args.csv)
else:
    detections = load_from_db()

print(f"Loaded {len(detections)} detection records")

if not detections:
    print("No detections found. Cannot compute percentiles.")
    sys.exit(1)

results = compute_camera_percentiles(detections, args.output)

# Print summary
print(f"\n{'='*60}")
print(f"  CAMERA PERCENTILE SUMMARY")
print(f"{'='*60}")

if "__global__" in results:
    g = results["__global__"]
    n = g['n_observations']
    p25 = g['p25_vehicles']; p75 = g['p75_vehicles']
    p95 = g['p95_vehicles']; mean = g['mean_vehicles']
    print(f"  Global ({n} observations):")
    print(f"    Vehicles:    P25={p25:.1f}  P75={p75:.1f}  P95={p95:.1f}  Mean={mean:.1f}")
    pp25 = g['p25_pedestrians']; pp75 = g['p75_pedestrians']
    pp95 = g['p95_pedestrians']; pmean = g['mean_pedestrians']
    print(f"    Pedestrians: P25={pp25:.1f}  P75={pp75:.1f}  P95={pp95:.1f}  Mean={pmean:.1f}")

print(f"\n  Per-camera (top 10 by observations):")
sorted_cams = sorted(
    [(k, v) for k, v in results.items() if k != "__global__"],
    key=lambda x: x[1]["n_observations"],
    reverse=True
)
for cam_id, stats in sorted_cams[:10]:
    n = stats['n_observations']
    p25 = stats['p25_vehicles']; p75 = stats['p75_vehicles']; p95 = stats['p95_vehicles']
    print(f"    {cam_id[:40]:40s}  n={n:4d}  P25={p25:4.1f}  P75={p75:4.1f}  P95={p95:4.1f}")

print(f"\n  Thresholds meaning:")
print(f"    LOW    = vehicles <= P25 (free flow)")
print(f"    MEDIUM = P25 < vehicles <= P75 (stable flow)")
print(f"    HIGH   = vehicles > P75 (congested)")
print(f"    Score  = (vehicles / P95) * 100 (normalised occupancy)")
print(f"\n  Output: {args.output}")
print(f"{'='*60}")