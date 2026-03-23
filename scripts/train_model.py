"""
RQ2: Fine-tune YOLOv8 on Newcastle traffic camera images
"""
from ultralytics import YOLO

# Load pre-trained model (same one used for RQ1 baseline)
model = YOLO("yolov8n.pt")

# Train on labeled Newcastle data
results = model.train(
    data="data/training/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="newcastle_traffic",
    patience=10,
    save=True
)

print("\nTraining complete!")
print(f"Best model saved to: runs/detect/newcastle_traffic/weights/best.pt")