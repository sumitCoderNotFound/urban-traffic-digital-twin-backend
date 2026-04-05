import yaml
from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path("data/Newcastle-Traffic-Detection.v4i.yolov8").resolve()

with open(DATASET_DIR / "data.yaml") as f:
    config = yaml.safe_load(f)

config['train'] = str(DATASET_DIR / "train" / "images")
config['val'] = str(DATASET_DIR / "valid" / "images")

fixed_yaml = DATASET_DIR / "data_improved.yaml"
with open(fixed_yaml, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Classes: {config['nc']} - {config['names']}")

model = YOLO("yolov8s.pt")
model.train(
    data=str(fixed_yaml),
    epochs=100,
    patience=20,
    imgsz=960,
    batch=4,
    device="cpu",
    name="newcastle_v4_improved",
    lr0=0.0005,
    lrf=0.01,
    freeze=10,
    mosaic=1.0,
    mixup=0.15,
    save=True,
    plots=True,
)
print("DONE! Best model: runs/detect/newcastle_v4_improved/weights/best.pt")
