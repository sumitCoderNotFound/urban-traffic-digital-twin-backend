"""
API endpoint to return annotated images with YOLO bounding boxes
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO
import cv2
import glob
import os

router = APIRouter(prefix="/api/annotate", tags=["Annotate"])
model = YOLO("yolov8n.pt")

@router.get("/cameras")
async def list_images():
    """List all available images in dataset"""
    images = sorted(glob.glob("data/dataset/*.jpg"))
    return [os.path.basename(img) for img in images]

@router.get("/detect/{filename}")
async def detect_image(filename: str, confidence: float = 0.4):
    """Run YOLO on an image and return annotated result"""
    filepath = f"data/dataset/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    results = model(filepath, conf=confidence, verbose=False)
    annotated = results[0].plot()
    
    _, buffer = cv2.imencode('.jpg', annotated)
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/summary/{filename}")
async def detection_summary(filename: str, confidence: float = 0.4):
    """Run YOLO and return detection details (no image)"""
    filepath = f"data/dataset/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    results = model(filepath, conf=confidence, verbose=False)
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": box.xyxy[0].tolist()
        })
    
    return {
        "filename": filename,
        "confidence_threshold": confidence,
        "total_objects": len(detections),
        "detections": detections
    }