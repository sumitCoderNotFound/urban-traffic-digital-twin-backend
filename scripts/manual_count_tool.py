"""
Manual Vehicle Count Validation Tool
=====================================
Opens each image, shows model count, you enter manual count.
Press Enter to skip, q to quit and save progress.

Run from: urban-digital-twin-backend/
Usage: python scripts/manual_count_tool.py

Author: Sumit Malviya (W24041293)
"""

import pandas as pd
import subprocess
import sys
import os

CSV_PATH = 'data/manual_validation/sample_frames.csv'

df = pd.read_csv(CSV_PATH)

# Only process unvalidated rows
todo = df[df['manual_vehicle_count'].isna()].copy()
done = df[df['manual_vehicle_count'].notna()].copy()

print(f"\n{'='*55}")
print(f"  MANUAL VEHICLE COUNT VALIDATION")
print(f"{'='*55}")
print(f"  Total frames:     {len(df)}")
print(f"  Already counted:  {len(done)}")
print(f"  Remaining:        {len(todo)}")
print(f"\n  Instructions:")
print(f"  - Image will open automatically")
print(f"  - Count ALL vehicles (cars, buses, trucks, motorcycles)")
print(f"  - Enter the number and press Enter")
print(f"  - Press Enter with no number to SKIP")
print(f"  - Type q to quit and save progress")
print(f"{'='*55}\n")

input("Press Enter to start...")

saved = 0

for i, row in todo.iterrows():
    img_path = row['image_path_full']
    frame_id = row['frame_id']
    camera = row['camera']
    hour = row['hour']
    period = row['period']
    model_count = int(row['model_vehicle_count'])

    # Open image
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', img_path])
        elif sys.platform == 'linux':
            subprocess.Popen(['eog', img_path])
    except Exception:
        pass

    print(f"\n[{frame_id:2d}/65] Camera: {camera} | Hour: {hour}:00 | Period: {period}")
    print(f"       Model detected: {model_count} vehicles")
    print(f"       Image: {img_path.split('/')[-1]}")

    user_input = input("       Your count (Enter=skip, q=quit): ").strip()

    if user_input.lower() == 'q':
        print("\nQuitting and saving progress...")
        break

    if user_input == '':
        print("       Skipped.")
        continue

    try:
        manual_count = int(user_input)
        df.at[i, 'manual_vehicle_count'] = manual_count
        diff = manual_count - model_count
        sign = "+" if diff >= 0 else ""
        print(f"       Saved: {manual_count} (model={model_count}, diff={sign}{diff})")
        saved += 1
    except ValueError:
        print("       Invalid input, skipped.")
        continue

# Save progress
df.to_csv(CSV_PATH, index=False)
print(f"\n{'='*55}")
print(f"  Saved {saved} new counts to {CSV_PATH}")

# Show stats for completed counts
completed = df[df['manual_vehicle_count'].notna()]
print(f"  Total completed: {len(completed)}/65")

if len(completed) >= 10:
    import numpy as np
    manual = completed['manual_vehicle_count'].values
    model = completed['model_vehicle_count'].values
    mae = np.mean(np.abs(manual - model))
    correlation = np.corrcoef(manual, model)[0,1]
    print(f"\n  Current validation stats:")
    print(f"  MAE:         {mae:.2f} vehicles")
    print(f"  Correlation: {correlation:.3f}")

print(f"{'='*55}")
print(f"\n  Run again to continue counting remaining frames.")
