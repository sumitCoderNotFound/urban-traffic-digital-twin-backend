import os, glob
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
os.makedirs("data/annotated", exist_ok=True)

images = sorted(glob.glob("data/dataset/*.jpg"))

print(f"Found {len(images)} images\n")
print("=== DAYTIME IMAGES ONLY (10am-4pm) ===\n")

for img_path in images:
    filename = os.path.basename(img_path)
    # Filter: only images between 10:00 and 16:00
    parts = filename.rsplit("_", 1)
    if len(parts) < 2:
        continue
    time_str = parts[-1].replace(".jpg", "")
    hour = int(time_str[:2]) if time_str[:2].isdigit() else -1
    if hour < 10 or hour > 16:
        continue

    output_path = f"data/annotated/DAY_{filename}"
    results = model(img_path, conf=0.4, verbose=False)

    for result in results:
        boxes = result.boxes
        import cv2
        cv2.imwrite(output_path, result.plot())

        classes = []
        for box in boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            classes.append(f"{name}({conf:.2f})")

        if classes:
            print(f"✅ {filename}: {len(boxes)} objects")
            for c in classes:
                print(f"   {c}")
            print(f"   Saved: {output_path}\n")
        else:
            print(f"⬚ {filename}: Nothing detected")

print("\n✅ Check data/annotated/ for DAY_ images")