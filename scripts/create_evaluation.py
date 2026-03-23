"""
Create RQ1 evaluation spreadsheet
Opens each annotated image so you can count objects and compare
"""
import os, glob, csv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
TARGET = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Get daytime images only
images = sorted(glob.glob("data/dataset/*_1[0-6]*.jpg"))[:50]

rows = []
for img_path in images:
    filename = os.path.basename(img_path)
    results = model(img_path, conf=0.4, verbose=False)
    
    cars = buses = trucks = persons = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 2: cars += 1
        elif cls_id == 5: buses += 1
        elif cls_id == 7: trucks += 1
        elif cls_id == 0: persons += 1
    
    rows.append({
        "filename": filename,
        "yolo_cars": cars,
        "yolo_buses": buses,
        "yolo_trucks": trucks,
        "yolo_persons": persons,
        "human_cars": "",
        "human_buses": "",
        "human_trucks": "",
        "human_persons": "",
        "notes": ""
    })

with open("data/rq1_evaluation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Created data/rq1_evaluation.csv with {len(rows)} images")
print("Open each annotated image in data/annotated/ and fill in the human_ columns")