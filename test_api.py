import requests
import warnings
import cv2
import numpy as np
from ultralytics import YOLO
warnings.filterwarnings('ignore')

print("Step 1: Getting camera images...\n")

# Get all cameras
r = requests.get("https://portal.cctv.urbanobservatory.ac.uk/latest", verify=False)
cameras = r.json()
print(f"Total cameras: {len(cameras)}")

model = YOLO("yolov8n.pt")
vehicles = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

for i, cam in enumerate(cameras):
    safe_path = cam['safe_photo_path']
    url = f"https://portal.cctv.urbanobservatory.ac.uk/photo/{safe_path}"
    
    img_response = requests.get(url, verify=False, timeout=10)
    if img_response.status_code != 200:
        continue
    
    img_path = f"camera_{i}.jpg"
    with open(img_path, "wb") as f:
        f.write(img_response.content)
    
    # Run YOLOv8
    results = model(img_path, conf=0.25, verbose=False)
    
    counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls in vehicles:
                counts[vehicles[cls]] += 1
    
    total = sum(counts.values())
    
    if total > 0:
        print(f"\n🎉 VEHICLES FOUND!")
        print(f"Camera: {cam['place']}")
        print(f"Timestamp: {cam['timestamp']}")
        print(f"Detections: {counts}")
        print(f"Total vehicles: {total}")
        
        # Save annotated image using plot()
        annotated = results[0].plot()
        cv2.imwrite("annotated_result.jpg", annotated)
        print(f"✅ Annotated image saved as annotated_result.jpg")
        break
    else:
        print(f"Camera {i} ({cam['place']}): no vehicles")

print("\n✅ Done!")