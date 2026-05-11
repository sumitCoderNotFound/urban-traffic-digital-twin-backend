"""
Run from your urban-digital-twin-backend folder:
    python3 scripts/crop_figure_titles.py

Crops the top title bar off figures that have "Fig. X" baked in.
Saves clean versions to figures/final/ with _clean suffix.
"""
from PIL import Image
import os

files_to_crop = [
    # (input file, output file, pixels to crop from top)
    ('figures/final/fig_03_detection_grid_3x3.png',   'figures/final/fig_05_detection_grid_clean.png',  55),
    ('figures/final/fig_04b_success_vs_failure.png',   'figures/final/fig_06_success_failure_clean.png', 55),
    ('figures/final/fig_11_training_curves_real.png',  'figures/final/fig_03_training_curves_clean.png', 55),
]

for src, dst, crop_top in files_to_crop:
    if os.path.exists(src):
        img = Image.open(src)
        w, h = img.size
        # Crop: left, top, right, bottom
        cropped = img.crop((0, crop_top, w, h))
        cropped.save(dst, dpi=(300, 300))
        print(f"✅ Cropped {crop_top}px from top: {dst}")
        print(f"   Original: {w}x{h} → Cropped: {w}x{h-crop_top}")
    else:
        print(f"❌ Not found: {src}")

print("\nDone. Use the _clean versions in your paper instead.")
