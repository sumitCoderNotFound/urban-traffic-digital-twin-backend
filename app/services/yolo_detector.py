import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class DetectionResult:
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

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.YOLO_MODEL
        self.model = None
        self.confidence_threshold = settings.YOLO_CONFIDENCE
        self.device = settings.YOLO_DEVICE
        self.class_map = settings.DETECTION_CLASSES

    def load_model(self) -> bool:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"YOLO model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False

    def is_available(self) -> bool:
        if self.model is None:
            return self.load_model()
        return True

    def detect(self, image_path: str) -> Optional[DetectionResult]:
        if not self.is_available():
            return None

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        start_time = time.time()

        try:
            results = self.model(
                image_path,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            counts = {
                "vehicles": 0, "pedestrians": 0, "cyclists": 0,
                "cars": 0, "buses": 0, "trucks": 0, "motorcycles": 0,
            }

            all_detections = []
            confidences = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    confidences.append(confidence)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    all_detections.append({
                        "class_id": class_id,
                        "class_name": self.class_map.get(class_id, "unknown"),
                        "confidence": round(confidence, 3),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)]
                    })

                    if class_id == 0:
                        counts["pedestrians"] += 1
                    elif class_id == 1:
                        counts["cyclists"] += 1
                    elif class_id == 2:
                        counts["vehicles"] += 1
                        counts["cars"] += 1
                    elif class_id == 3:
                        counts["vehicles"] += 1
                        counts["motorcycles"] += 1
                    elif class_id == 5:
                        counts["vehicles"] += 1
                        counts["buses"] += 1
                    elif class_id == 7:
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
            print(f"Detection failed: {e}")
            return None

    def save_annotated_image(self, image_path: str, output_path: str) -> bool:
        if not self.is_available():
            return False
        try:
            results = self.model(image_path, conf=self.confidence_threshold, device=self.device, verbose=False)
            for result in results:
                import cv2
                cv2.imwrite(output_path, result.plot())
            return True
        except Exception as e:
            print(f"Failed to save annotated image: {e}")
            return False


_detector_instance = None


def get_detector() -> YOLODetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector()
    return _detector_instance