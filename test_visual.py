"""
Visual test: Run YOLOv8 and save images WITH bounding boxes drawn
Also tests with lower confidence to see what the model picks up at night
"""
import os
import glob
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
os.makedirs("data/annotated", exist_ok=True)

# Get all images
images = sorted(glob.glob("data/dataset/*.jpg"))

if not images:
    print("No images found in data/dataset/")
    exit()

print(f"Found {len(images)} images\n")

for img_path in images[:10]:  # Test first 10
    filename = os.path.basename(img_path)
    output_path = f"data/annotated/DETECTED_{filename}"

    # Run detection with LOWER confidence (0.25) to see more
    results = model(img_path, conf=0.25, verbose=False)

    for result in results:
        # Count detections
        boxes = result.boxes
        count = len(boxes)

        # Save image with bounding boxes drawn on it
        annotated = result.plot()  # This draws the boxes

        import cv2
        cv2.imwrite(output_path, annotated)

        # Print what was found
        classes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            classes.append(f"{name}({conf:.2f})")

        status = ", ".join(classes) if classes else "Nothing detected"
        print(f"{filename}: {count} objects -> {status}")
        print(f"  Saved: {output_path}")

print(f"\n✅ Open data/annotated/ folder in Finder to see results")