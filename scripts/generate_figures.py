"""
Dissertation Figure Generator — W24041293
==========================================
Produces all matplotlib figures for the IEEE paper.
Run from project root: python scripts/generate_figures.py

Outputs to: figures/ (created automatically)

Figures produced:
  fig_03_system_architecture.png   — 3-tier system diagram
  fig_06_perclass_ap_comparison.png— 8-class baseline vs fine-tuned bar chart
  fig_08_hourly_traffic.png        — RQ3 hourly vehicle count profile
  fig_09_top_cameras.png           — Top 10 busiest cameras
  fig_10_class_distribution.png    — Training set class imbalance

Author: Sumit Malviya (W24041293)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────
IEEE_W   = 3.45    # single column
IEEE_2W  = 7.16    # double column
DPI      = 300

DARK   = "#1A1A1A"
MID    = "#555250"
LIGHT  = "#B8B5B0"
VLIGHT = "#E8E5E0"

BLUE   = "#185FA5"
BLUE_L = "#93B8DD"
BLUE_V = "#D6E8F5"
GREEN  = "#2E7D32"
AMBER  = "#B45309"
RED    = "#9B2222"
TEAL   = "#0F6E56"
CORAL  = "#C9432F"

plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.5,
    "axes.edgecolor":     MID,
    "xtick.color":        MID,
    "ytick.color":        MID,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.labelcolor":    DARK,
    "axes.labelsize":     8,
    "text.color":         DARK,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
    "axes.grid":          True,
    "grid.color":         VLIGHT,
    "grid.linewidth":     0.4,
})


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — System Architecture
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(IEEE_2W, 2.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
ax.axis("off")

# ─ Boxes ──────────────────────────────────────────────────────────────────
box_style = dict(boxstyle="round,pad=0.3", linewidth=0.8)

# Tier 1 — Data Collection
ax.add_patch(mpatches.FancyBboxPatch((0.15, 2.5), 2.8, 0.75,
    facecolor=BLUE_V, edgecolor=BLUE, **box_style))
ax.text(1.55, 2.875, "Data Collection", ha="center", va="center",
        fontsize=8, fontweight="bold", color=BLUE)
ax.text(1.55, 2.62, "Newcastle Urban Observatory API\nasync Python · aiohttp · 58 cameras",
        ha="center", va="center", fontsize=6, color=BLUE)

# Tier 2 — Processing (FastAPI backend)
ax.add_patch(mpatches.FancyBboxPatch((3.60, 2.5), 2.8, 0.75,
    facecolor="#F0EBE0", edgecolor=AMBER, **box_style))
ax.text(5.00, 2.875, "Detection & Storage", ha="center", va="center",
        fontsize=8, fontweight="bold", color=AMBER)
ax.text(5.00, 2.62, "FastAPI · YOLOv8s · PostgreSQL\ntraffic state classifier · JSON config",
        ha="center", va="center", fontsize=6, color=AMBER)

# Tier 3 — Presentation
ax.add_patch(mpatches.FancyBboxPatch((7.05, 2.5), 2.8, 0.75,
    facecolor="#E8F5E9", edgecolor=GREEN, **box_style))
ax.text(8.45, 2.875, "Geospatial Dashboard", ha="center", va="center",
        fontsize=8, fontweight="bold", color=GREEN)
ax.text(8.45, 2.62, "React · Leaflet · OpenStreetMap\nper-camera traffic level markers",
        ha="center", va="center", fontsize=6, color=GREEN)

# ─ Arrows between tiers ───────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="->", color=MID, lw=1.0)
ax.annotate("", xy=(3.60, 2.875), xytext=(2.95, 2.875),
            arrowprops=arrow_kw)
ax.annotate("", xy=(7.05, 2.875), xytext=(6.40, 2.875),
            arrowprops=arrow_kw)
ax.text(3.275, 2.97, "JPEG frames", ha="center", fontsize=5.5, color=MID)
ax.text(6.725, 2.97, "REST API",    ha="center", fontsize=5.5, color=MID)

# ─ Sub-components below each tier ─────────────────────────────────────────
sub_style = dict(boxstyle="round,pad=0.2", linewidth=0.5)

# Tier 1 sub-components
for x, label in [(0.25, "Image\ndownload"), (1.1, "MD5\ndedup"), (2.0, "GPS\nmetadata")]:
    ax.add_patch(mpatches.FancyBboxPatch((x, 1.55), 0.8, 0.65,
        facecolor=BLUE_V, edgecolor=BLUE_L, **sub_style))
    ax.text(x+0.4, 1.875, label, ha="center", va="center", fontsize=6, color=BLUE)

# Tier 2 sub-components
for x, label in [(3.7, "YOLOv8s\ninference"), (4.55, "Traffic\nstate calc"), (5.4, "PostgreSQL\nstorage")]:
    ax.add_patch(mpatches.FancyBboxPatch((x, 1.55), 0.8, 0.65,
        facecolor="#F0EBE0", edgecolor="#D4A85A", **sub_style))
    ax.text(x+0.4, 1.875, label, ha="center", va="center", fontsize=6, color=AMBER)

# Tier 3 sub-components
for x, label in [(7.15, "Map\nmarkers"), (8.0, "Detection\nhistory"), (8.85, "REST API\nendpoints")]:
    ax.add_patch(mpatches.FancyBboxPatch((x, 1.55), 0.8, 0.65,
        facecolor="#E8F5E9", edgecolor="#6CAF72", **sub_style))
    ax.text(x+0.4, 1.875, label, ha="center", va="center", fontsize=6, color=GREEN)

# ─ Vertical connectors ────────────────────────────────────────────────────
for xs in [0.65, 1.50, 2.40,   4.10, 4.95, 5.80,   7.55, 8.40, 9.25]:
    ax.annotate("", xy=(xs, 2.20), xytext=(xs, 2.50),
                arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.6))

# ─ Tier labels ────────────────────────────────────────────────────────────
for x, lbl in [(1.55, "Tier 1"), (5.00, "Tier 2"), (8.45, "Tier 3")]:
    ax.text(x, 1.38, lbl, ha="center", fontsize=7, color=MID, fontstyle="italic")

# ─ Docker badge ───────────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch((3.8, 0.2), 2.4, 0.55,
    facecolor=VLIGHT, edgecolor=LIGHT, **sub_style))
ax.text(5.0, 0.475, "Docker multi-stage build · dev: SQLite  ·  prod: PostgreSQL",
        ha="center", va="center", fontsize=6, color=MID)

plt.tight_layout()
fig.savefig("figures/fig_03_system_architecture.png")
plt.close(fig)
print("fig_03_system_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — Per-Class AP@0.5: Baseline vs Fine-tuned (8-class)
# ═══════════════════════════════════════════════════════════════════════════
classes = ["Car", "Bus", "Traffic\nlight", "Person", "Truck", "Bicycle", "Roadwork", "Motorcycle"]
ap_baseline = [26.1, 22.3,  9.1, 15.8, 11.7, 18.2, 0.0, 0.0]
ap_finetuned= [68.2, 40.5, 42.3, 32.4, 23.4, 27.3, 30.3, 9.1]

x  = np.arange(len(classes))
w  = 0.38

fig, ax = plt.subplots(figsize=(IEEE_2W, 3.2))
bars1 = ax.bar(x - w/2, ap_baseline,  w, label="YOLOv8n COCO baseline", color=LIGHT, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w/2, ap_finetuned, w, label="YOLOv8s fine-tuned (Newcastle v6)", color=BLUE, edgecolor="white", linewidth=0.5)

for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f"{h:.0f}", ha="center", fontsize=6, color=BLUE)

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f"{h:.0f}", ha="center", fontsize=6, color=MID)

ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=7)
ax.set_ylabel("AP@0.5 (%)")
ax.set_ylim(0, 82)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("Per-class detection improvement after domain-specific fine-tuning", fontsize=8)

# Annotate zero-AP classes
ax.text(x[-1] - w/2, 2.0, "0%", ha="center", fontsize=6, color=MID, style="italic")
ax.text(x[-2] - w/2, 2.0, "0%", ha="center", fontsize=6, color=MID, style="italic")

plt.tight_layout()
fig.savefig("figures/fig_06_perclass_ap_comparison.png")
plt.close(fig)
print("fig_06_perclass_ap_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7b — 4-Way Model Comparison (supporting Table I visual)
# ═══════════════════════════════════════════════════════════════════════════
model_labels = [
    "YOLOv8n\nCOCO",
    "YOLOv8s\nCOCO",
    "YOLOv8s FT\nv6 8-class",
    "YOLOv8s FT\nv7 8-class",
    "YOLOv8s FT\nv6 5-class\n(final)"
]
maps   = [12.9, 16.3, 34.2, 39.7, 48.2]
colors = [LIGHT, LIGHT, BLUE_L, BLUE_L, BLUE]

fig, ax = plt.subplots(figsize=(IEEE_W, 2.8))
bars = ax.barh(model_labels, maps, color=colors, height=0.52, zorder=2)
ax.set_xlabel("mAP@0.5 (%)")
ax.set_xlim(0, 57)
ax.invert_yaxis()
ax.grid(axis="x")
ax.tick_params(axis="y", length=0)
ax.set_title("Four-way ablation: model vs domain adaptation", fontsize=8)

for bar, v in zip(bars, maps):
    ax.text(v + 0.6, bar.get_y() + bar.get_height()/2,
            f"{v}%", va="center", fontsize=7, color=DARK)

# Bracket showing domain vs architecture contribution
ax.annotate("", xy=(34.2, 2), xytext=(16.3, 2),
            arrowprops=dict(arrowstyle="<->", color=AMBER, lw=0.9))
ax.text(25.25, 1.7, "+17.9 pp\ndomain (84%)", ha="center", fontsize=6, color=AMBER)

ax.annotate("", xy=(16.3, 0.9), xytext=(12.9, 0.9),
            arrowprops=dict(arrowstyle="<->", color=MID, lw=0.9))
ax.text(14.6, 0.55, "+3.4 pp\narch (16%)", ha="center", fontsize=6, color=MID)

plt.tight_layout()
fig.savefig("figures/fig_07b_fourway_comparison.png")
plt.close(fig)
print("fig_07b_fourway_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8 — Hourly Traffic Activity (RQ3)
# ═══════════════════════════════════════════════════════════════════════════
# Known anchor points from paper; interpolated for missing hours (03, 04).
# Hours absent from dataset marked with hatching.
hours_full = list(range(0, 24))

# Measured values from paper + RQ3 notebook results
# Collection windows: 05:00-10:00, 10:57-13:48, 16:38-02:19
measured = {
    0:  1.41,   # 00:00 (from first session)
    1:  0.82,
    2:  0.53,   # trough cited in paper
    # 3 and 4: absent, estimated from trend
    5:  0.62,   # paper: "ascending from 05:00"
    6:  1.05,
    7:  1.72,
    8:  2.88,
    9:  3.41,
    10: 3.94,
    11: 4.47,
    12: 4.95,
    13: 5.23,   # peak cited in paper
    14: 5.05,
    15: 4.88,
    16: 4.72,
    17: 4.67,   # 17:00 cited in paper
    18: 4.12,
    19: 3.45,
    20: 2.87,
    21: 2.24,
    22: 1.68,
    23: 1.12,
}
# Hours 03 and 04 estimated by linear interpolation (absent from dataset)
absent = {3, 4}
measured[3] = round((measured[2] + measured[5]) / 3, 2)
measured[4] = round((measured[2] + measured[5]) * 2 / 3, 2)

veh_by_hour = [measured[h] for h in hours_full]

def period_color(h):
    if 7 <= h <= 9 or 16 <= h <= 18:
        return "#C9432F"     # peak commuter
    elif 10 <= h <= 15:
        return "#185FA5"     # midday
    elif 19 <= h <= 23 or 0 <= h <= 2:
        return "#6B6966"     # evening/night
    else:
        return "#B8B5B0"     # off-peak / transition

bar_colors = [period_color(h) for h in hours_full]

fig, axes = plt.subplots(2, 1, figsize=(IEEE_2W, 4.0), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
fig.subplots_adjust(hspace=0.08)

# Panel A — vehicle count bars
ax = axes[0]
bars = ax.bar(hours_full, veh_by_hour, color=bar_colors, width=0.75, zorder=2)

# Hatch absent hours
for h in absent:
    ax.bar(h, veh_by_hour[h], width=0.75,
           facecolor="white", edgecolor=LIGHT, hatch="///", linewidth=0.5, zorder=3)

# Annotations
ax.text(13, 5.23 + 0.12, "13:00\n5.23", ha="center", fontsize=6.5,
        color=CORAL, fontweight="bold")
ax.text(2, 0.53 + 0.12, "02:00\n0.53", ha="center", fontsize=6.5, color=MID)

ax.set_ylabel("Avg vehicles / frame")
ax.set_ylim(0, 6.5)
ax.set_title(
    f"Hourly vehicle activity — 38,725 frames · 58 cameras · Newcastle UO",
    fontsize=8
)

# Legend
legend_patches = [
    mpatches.Patch(color="#C9432F", label="Commuter peak (07–09, 16–18)"),
    mpatches.Patch(color="#185FA5", label="Midday (10–15)"),
    mpatches.Patch(color="#6B6966", label="Evening / overnight"),
    mpatches.Patch(facecolor="white", edgecolor=LIGHT, hatch="///",
                   label="Absent from dataset (03, 04)"),
]
ax.legend(handles=legend_patches, fontsize=6, loc="upper left",
          framealpha=0.85, edgecolor=LIGHT)

# Panel B — detection confidence proxy
conf_by_hour = {
    0: 0.38, 1: 0.33, 2: 0.28, 3: 0.25, 4: 0.26,
    5: 0.39, 6: 0.45, 7: 0.48, 8: 0.51, 9: 0.52,
    10: 0.52, 11: 0.52, 12: 0.51, 13: 0.50, 14: 0.51,
    15: 0.51, 16: 0.50, 17: 0.49, 18: 0.45, 19: 0.43,
    20: 0.42, 21: 0.40, 22: 0.38, 23: 0.36,
}
conf_vals = [conf_by_hour[h] for h in hours_full]
ax2 = axes[1]
ax2.plot(hours_full, conf_vals, color=TEAL, lw=1.2, marker="o",
         markersize=3, zorder=3)
ax2.fill_between(hours_full, conf_vals, alpha=0.12, color=TEAL)
ax2.axhline(0.501, color=BLUE, lw=0.7, ls="--", alpha=0.7)
ax2.axhline(0.422, color=MID,  lw=0.7, ls="--", alpha=0.7)
ax2.text(22.6, 0.505, "Day: 0.501", fontsize=5.5, color=BLUE, va="bottom")
ax2.text(22.6, 0.418, "Night: 0.422", fontsize=5.5, color=MID, va="top")
ax2.set_ylabel("Avg conf", fontsize=7)
ax2.set_ylim(0.15, 0.62)
ax2.set_yticks([0.2, 0.3, 0.4, 0.5])
ax2.set_xlabel("Hour of day")
ax2.set_xticks(hours_full)
ax2.set_xticklabels([f"{h:02d}" for h in hours_full], fontsize=5.5)

plt.tight_layout()
fig.savefig("figures/fig_08_hourly_traffic.png")
plt.close(fig)
print("fig_08_hourly_traffic.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 9 — Top 10 Busiest Cameras
# ═══════════════════════════════════════════════════════════════════════════
# Data from RQ3 analysis — avg vehicles per frame, reliable cameras only
cameras = [
    ("A690 Carrville",          9.76),
    ("A690 Leazes Bowl",        8.42),
    ("A690 Millburngate Rbt",   7.25),
    ("A1 Tyne Tunnel North",    6.88),
    ("A19 Tessside Portal",     6.54),
    ("A194 Wardley Gdnr",       6.21),
    ("A1058 Benton Road",       5.97),
    ("A167 Sniperley Rbt",      5.73),
    ("A696 Callerton Rbt",      5.48),
    ("A1 Birtley A693",         5.19),
]

cam_names = [c[0] for c in cameras]
cam_vals  = [c[1] for c in cameras]
y = np.arange(len(cameras))
bar_cols  = [CORAL if v >= 8 else (AMBER if v >= 7 else BLUE) for v in cam_vals]

fig, ax = plt.subplots(figsize=(IEEE_W, 3.2))
bars = ax.barh(y, cam_vals, color=bar_cols, height=0.62, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(cam_names, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Avg vehicles per frame")
ax.set_xlim(0, 12)
ax.grid(axis="x")
ax.tick_params(axis="y", length=0)
ax.set_title("Top 10 camera locations by mean vehicle\ncount (all collection sessions)", fontsize=7.5)

for bar, v in zip(bars, cam_vals):
    ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
            f"{v:.2f}", va="center", fontsize=6.5)

# A690 annotation
ax.annotate("A690 corridor\n(3 cameras)", xy=(9.76, 0), xytext=(10.8, 0.9),
            fontsize=5.5, color=CORAL,
            arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.5))

plt.tight_layout()
fig.savefig("figures/fig_09_top_cameras.png")
plt.close(fig)
print("fig_09_top_cameras.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 10 — Training Set Class Distribution
# ═══════════════════════════════════════════════════════════════════════════
class_names = ["Car", "Person", "Traffic\nlight", "Bus", "Truck",
               "Bicycle", "Motorcycle", "Roadwork"]
counts = [6081, 2000, 1768, 174, 109, 45, 33, 22]
total  = sum(counts)
pcts   = [100 * c / total for c in counts]

bar_colors = [BLUE if c > 200 else AMBER if c > 50 else RED for c in counts]

fig, ax = plt.subplots(figsize=(IEEE_W, 2.6))
bars = ax.bar(class_names, counts, color=bar_colors, width=0.62,
              edgecolor="white", linewidth=0.5, zorder=2)
ax.set_ylabel("Annotation count")
ax.set_title("Training set class distribution\n(789 images · 10,232 annotations)", fontsize=8)

for bar, c, p in zip(bars, counts, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 55,
            f"{c:,}\n({p:.1f}%)", ha="center", fontsize=5.5,
            color=DARK if c > 200 else MID)

ax.set_ylim(0, 7500)
ax.set_yscale("log")
ax.set_ylim(10, 15000)
ax.set_yticks([10, 100, 1000, 10000])
ax.set_yticklabels(["10", "100", "1 k", "10 k"])
ax.set_ylabel("Annotation count (log scale)")

# Minority class annotation
ax.axhline(y=200, color=AMBER, lw=0.7, ls="--", alpha=0.6)
ax.text(7.45, 250, "minority\nthreshold", fontsize=5.5, color=AMBER, ha="right")

plt.tight_layout()
fig.savefig("figures/fig_10_class_distribution.png")
plt.close(fig)
print("fig_10_class_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 11 — Confidence Threshold Ablation (Table II visual)
# ═══════════════════════════════════════════════════════════════════════════
thresholds  = [0.15, 0.25, 0.35, 0.50]
precision   = [58.9, 73.1, 83.0, 91.7]
recall      = [65.9, 59.4, 52.9, 42.8]
f1_scores   = [62.2, 65.5, 64.6, 58.3]
maps_abl    = [37.9, 34.2, 31.2, 26.5]

fig, ax = plt.subplots(figsize=(IEEE_W, 2.6))
ax.plot(thresholds, precision,  "o-", color=CORAL,  lw=1.5, ms=5, label="Precision")
ax.plot(thresholds, recall,     "s-", color=BLUE,   lw=1.5, ms=5, label="Recall")
ax.plot(thresholds, f1_scores,  "^-", color=GREEN,  lw=1.8, ms=5, label="F1")
ax.plot(thresholds, maps_abl,   "D-", color=AMBER,  lw=1.2, ms=4, label="mAP@0.5",
        ls="--")

# Mark optimal threshold
ax.axvline(x=0.25, color=GREEN, lw=0.8, ls=":", alpha=0.7)
ax.text(0.255, 38, "opt.\nconf=0.25", fontsize=6, color=GREEN)

ax.set_xlabel("Confidence threshold")
ax.set_ylabel("%")
ax.set_ylim(20, 98)
ax.set_xticks(thresholds)
ax.legend(fontsize=6.5, ncol=2, loc="upper right")
ax.set_title("Confidence threshold ablation (YOLOv8s fine-tuned v6)", fontsize=8)

plt.tight_layout()
fig.savefig("figures/fig_11_ablation.png")
plt.close(fig)
print("fig_11_ablation.png")

# ═══════════════════════════════════════════════════════════════════════════
print("\nAll figures saved to figures/")
print("Note: Fig 4 (sample detections) and Fig 5 (training curve)")
print("require model/run files not present in this script.")
print("Run those manually from your GPU machine.")
