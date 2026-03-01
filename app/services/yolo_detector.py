"""
YOLO Detector Service
Object detection using YOLOv8 for traffic analysis
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass
class DetectionResult:
    """Result from YOLO detection"""
    vehicles: int
    pedestrians: int
    cyclists: int
    cars: int
    buses: int
    trucks: int
    motorcycles: int
    confidence_avg: float
    processing_time_ms: int
    detections: List[Dict]


class YOLODetector:
    """
    YOLOv8 detector for traffic analysis.
    
    Detects and counts vehicles, pedestrians, and cyclists
    from traffic camera images.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise the YOLO detector.
        
        Args:
            model_path: Path to YOLO model file. Defaults to settings.YOLO_MODEL
        """
        self.model_path = model_path or settings.YOLO_MODEL
        self.model = None
        self.confidence_threshold = settings.YOLO_CONFIDENCE
        self.device = settings.YOLO_DEVICE
        
        # Class mappings from COCO dataset
        self.class_map = settings.DETECTION_CLASSES
        
        # Vehicle classes
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        self.pedestrian_classes = {0}  # person
        self.cyclist_classes = {1}  # bicycle
    
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded: {self.model_path}")
            return True
        except ImportError:
            print("❌ ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if detector is ready to use"""
        if self.model is None:
            return self.load_model()
        return True
    
    def detect(self, image_path: str) -> Optional[DetectionResult]:
        """
        Run detection on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult with counts and details, or None if failed
        """
        if not self.is_available():
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                image_path,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            # Count detections
            counts = {
                "vehicles": 0,
                "pedestrians": 0,
                "cyclists": 0,
                "cars": 0,
                "buses": 0,
                "trucks": 0,
                "motorcycles": 0,
            }
            
            all_detections = []
            confidences = []
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    confidences.append(confidence)
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        "class_id": class_id,
                        "class_name": self.class_map.get(class_id, "unknown"),
                        "confidence": round(confidence, 3),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)]
                    }
                    all_detections.append(detection)
                    
                    # Count by category
                    if class_id == 0:  # person
                        counts["pedestrians"] += 1
                    elif class_id == 1:  # bicycle
                        counts["cyclists"] += 1
                    elif class_id == 2:  # car
                        counts["vehicles"] += 1
                        counts["cars"] += 1
                    elif class_id == 3:  # motorcycle
                        counts["vehicles"] += 1
                        counts["motorcycles"] += 1
                    elif class_id == 5:  # bus
                        counts["vehicles"] += 1
                        counts["buses"] += 1
                    elif class_id == 7:  # truck
                        counts["vehicles"] += 1
                        counts["trucks"] += 1
            
            processing_time = int((time.time() - start_time) * 1000)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return DetectionResult(
                vehicles=counts["vehicles"],
                pedestrians=counts["pedestrians"],
                cyclists=counts["cyclists"],
                cars=counts["cars"],
                buses=counts["buses"],
                trucks=counts["trucks"],
                motorcycles=counts["motorcycles"],
                confidence_avg=round(avg_confidence, 3),
                processing_time_ms=processing_time,
                detections=all_detections
            )
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return None
    
    def detect_from_url(self, image_url: str) -> Optional[DetectionResult]:
        """
        Run detection on an image from URL.
        
        Downloads the image temporarily and runs detection.
        
        Args:
            image_url: URL of the image
            
        Returns:
            DetectionResult or None if failed
        """
        import tempfile
        import requests
        
        try:
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            
            # Run detection
            result = self.detect(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to detect from URL: {e}")
            return None
    
    def save_annotated_image(
        self,
        image_path: str,
        output_path: str,
        show_labels: bool = True,
        show_conf: bool = True
    ) -> bool:
        """
        Save an annotated version of the image with bounding boxes.
        
        Args:
            image_path: Input image path
            output_path: Output path for annotated image
            show_labels: Show class labels
            show_conf: Show confidence scores
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            results = self.model(
                image_path,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            # Save annotated image
            for result in results:
                annotated = result.plot(labels=show_labels, conf=show_conf)
                import cv2
                cv2.imwrite(output_path, annotated)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save annotated image: {e}")
            return False


# Singleton instance
_detector_instance = None

def get_detector() -> YOLODetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector()
    return _detector_instance
