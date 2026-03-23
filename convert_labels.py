"""
Convert polygon labels to YOLO bounding box format
"""
import os, glob

label_dirs = [
    "data/training/train/labels",
    "data/training/valid/labels",
    "data/training/test/labels"
]

converted = 0
for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        continue
    for label_file in glob.glob(f"{label_dir}/*.txt"):
        new_lines = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = parts[0]
                coords = [float(x) for x in parts[1:]]
                
                # Extract x and y coordinates from polygon
                xs = coords[0::2]
                ys = coords[1::2]
                
                # Convert to bounding box
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                
                # YOLO format: center_x, center_y, width, height
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min
                
                new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        with open(label_file, "w") as f:
            f.write("\n".join(new_lines))
        converted += 1

print(f"✅ Converted {converted} label files to YOLO bounding box format")