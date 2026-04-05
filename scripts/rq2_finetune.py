import argparse
import os
import sys
import yaml
from pathlib import Path

# ── Parse arguments ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="RQ2: Fine-tune YOLOv8n on Newcastle CCTV")
parser.add_argument("--model",    default="yolov8n.pt",                    help="Base model weights")
parser.add_argument("--data-dir", default="data/ground_truth_roboflow",    help="Roboflow dataset directory")
parser.add_argument("--epochs",   type=int, default=50,                    help="Training epochs")
parser.add_argument("--batch",    type=int, default=8,                     help="Batch size (reduce to 4 if memory error)")
parser.add_argument("--imgsz",    type=int, default=640,                   help="Image size")
parser.add_argument("--device",   default="cpu",                           help="Device: cpu / mps / 0")
parser.add_argument("--patience", type=int, default=15,                    help="Early stopping patience")
parser.add_argument("--name",     default="newcastle_finetune",            help="Run name")
args = parser.parse_args()

# ── Import ultralytics ───────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# ── Fix data.yaml with absolute paths ────────────────────────
data_dir = Path(args.data_dir).resolve()
data_yaml = data_dir / "data.yaml"

if not data_yaml.exists():
    print(f"ERROR: data.yaml not found at {data_yaml}")
    sys.exit(1)

# Read existing data.yaml
with open(data_yaml, "r") as f:
    config = yaml.safe_load(f)

# Update with absolute paths
config["train"] = str(data_dir / "train" / "images")
config["val"] = str(data_dir / "valid" / "images")

# Check if test exists
test_dir = data_dir / "test" / "images"
if test_dir.exists():
    config["test"] = str(test_dir)

# Write updated data.yaml
updated_yaml = data_dir / "data_finetune.yaml"
with open(updated_yaml, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

# ── Verify dataset ───────────────────────────────────────────
train_imgs = list((data_dir / "train" / "images").glob("*"))
valid_imgs = list((data_dir / "valid" / "images").glob("*"))
train_lbls = list((data_dir / "train" / "labels").glob("*.txt"))
valid_lbls = list((data_dir / "valid" / "labels").glob("*.txt"))

print("=" * 60)
print("  RQ2: FINE-TUNING YOLOv8n ON NEWCASTLE CCTV")
print("=" * 60)
print(f"  Base model:     {args.model}")
print(f"  Dataset:        {data_dir}")
print(f"  Train images:   {len(train_imgs)}")
print(f"  Train labels:   {len(train_lbls)}")
print(f"  Valid images:   {len(valid_imgs)}")
print(f"  Valid labels:   {len(valid_lbls)}")
print(f"  Classes:        {config.get('nc', '?')} — {config.get('names', '?')}")
print(f"  Epochs:         {args.epochs}")
print(f"  Batch size:     {args.batch}")
print(f"  Image size:     {args.imgsz}")
print(f"  Device:         {args.device}")
print(f"  Patience:       {args.patience}")
print(f"  Data YAML:      {updated_yaml}")
print("=" * 60)

if len(train_imgs) == 0:
    print("ERROR: No training images found!")
    sys.exit(1)

if len(train_lbls) == 0:
    print("ERROR: No training labels found!")
    sys.exit(1)

# ── Load and train model ─────────────────────────────────────
print("\nLoading pre-trained YOLOv8n...")
model = YOLO(args.model)

print("Starting fine-tuning...\n")
results = model.train(
    data=str(updated_yaml),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    name=args.name,
    patience=args.patience,
    save=True,
    plots=True,
    verbose=True,
    
    # Hyperparameters optimised for small dataset fine-tuning
    lr0=0.001,           # Lower initial learning rate (default 0.01)
    lrf=0.01,            # Final learning rate factor
    warmup_epochs=5,     # Warm up slowly
    warmup_momentum=0.5,
    weight_decay=0.001,  # Regularisation to prevent overfitting
    
    # Augmentation (important for small dataset)
    hsv_h=0.015,         # Hue augmentation
    hsv_s=0.5,           # Saturation augmentation
    hsv_v=0.3,           # Value/brightness augmentation
    degrees=5.0,         # Small rotation
    translate=0.1,       # Translation
    scale=0.3,           # Scale augmentation
    flipud=0.0,          # No vertical flip (cars don't flip)
    fliplr=0.5,          # Horizontal flip
    mosaic=0.8,          # Mosaic augmentation
    mixup=0.1,           # Mixup augmentation
)

# ── Print results ────────────────────────────────────────────
best_model = Path(f"runs/detect/{args.name}/weights/best.pt")
last_model = Path(f"runs/detect/{args.name}/weights/last.pt")

print(f"\n{'=' * 60}")
print(f"  FINE-TUNING COMPLETE")
print(f"{'=' * 60}")
print(f"  Best model: {best_model}")
print(f"  Last model: {last_model}")
print(f"{'─' * 60}")
print(f"  NEXT STEPS:")
print(f"  1. Run RQ2 evaluation to compare before vs after:")
print(f"     python scripts/rq1_baseline_evaluation.py --model {best_model}")
print(f"  2. Compare against RQ1 baseline results")
print(f"  3. The improvement shows the value of domain adaptation")
print(f"{'=' * 60}")