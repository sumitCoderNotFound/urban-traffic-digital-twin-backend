from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.db.database import Base


class CameraStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TrafficLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    area = Column(String(100), nullable=True)
    status = Column(String(20), default=CameraStatus.ONLINE)
    is_active = Column(Boolean, default=True)
    uo_sensor_id = Column(String(100), nullable=True)
    uo_feed_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_image_at = Column(DateTime(timezone=True), nullable=True)

    detections = relationship("Detection", back_populates="camera", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Camera {self.id}: {self.name}>"


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    vehicles = Column(Integer, default=0)
    pedestrians = Column(Integer, default=0)
    cyclists = Column(Integer, default=0)
    cars = Column(Integer, default=0)
    buses = Column(Integer, default=0)
    trucks = Column(Integer, default=0)
    motorcycles = Column(Integer, default=0)
    traffic_level = Column(String(20), default=TrafficLevel.UNKNOWN)
    congestion_score = Column(Float, default=0.0)
    confidence_avg = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    image_path = Column(String(500), nullable=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    camera = relationship("Camera", back_populates="detections")

    def __repr__(self):
        return f"<Detection {self.id}: Camera {self.camera_id} at {self.detected_at}>"


class HourlyMetric(Base):
    __tablename__ = "hourly_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=True)
    hour_start = Column(DateTime(timezone=True), nullable=False)
    total_vehicles = Column(Integer, default=0)
    total_pedestrians = Column(Integer, default=0)
    total_cyclists = Column(Integer, default=0)
    avg_vehicles = Column(Float, default=0.0)
    avg_pedestrians = Column(Float, default=0.0)
    avg_cyclists = Column(Float, default=0.0)
    max_vehicles = Column(Integer, default=0)
    max_pedestrians = Column(Integer, default=0)
    avg_congestion_score = Column(Float, default=0.0)
    peak_traffic_level = Column(String(20), nullable=True)
    detection_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<HourlyMetric {self.hour_start}: {self.total_vehicles} vehicles>"


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), nullable=False)
    component = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<SystemLog {self.level}: {self.message[:50]}>"