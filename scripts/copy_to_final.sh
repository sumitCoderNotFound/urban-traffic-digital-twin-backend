#!/bin/bash
mkdir -p figures/final

# Generated figures
cp figures/fig_01_class_distribution.png figures/final/
cp figures/fig_02_system_architecture.png figures/final/
cp figures/fig_03_domain_gap.png figures/final/
cp figures/fig_04_training_curve.png figures/final/
cp figures/fig_05_threshold_ablation.png figures/final/
cp figures/fig_06_validation_scatter.png figures/final/
cp figures/fig_07_top_cameras.png figures/final/
cp figures/fig_08_traffic_state.png figures/final/

# From runs/notebooks (copy if they exist)
[ -f figures/fig_rq3_temporal_spatial.png ] && cp figures/fig_rq3_temporal_spatial.png figures/final/
[ -f figures/fig_03_detection_grid_3x3.png ] && cp figures/fig_03_detection_grid_3x3.png figures/final/
[ -f figures/fig_04b_success_vs_failure.png ] && cp figures/fig_04b_success_vs_failure.png figures/final/
[ -f figures/fig_rq1_domain_gap.png ] && cp figures/fig_rq1_domain_gap.png figures/final/
[ -f figures/fig_rq1_baseline_results.png ] && cp figures/fig_rq1_baseline_results.png figures/final/

echo "Files in figures/final/:"
ls figures/final/
