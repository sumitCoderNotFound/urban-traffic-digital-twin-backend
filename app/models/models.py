"""
Database Models
SQLAlchemy ORM models for Urban Digital Twin
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.db.database import Base


class CameraStatus(str, enum.Enum):
    """Camera status enum"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TrafficLevel(str, enum.Enum):
    """Traffic level enum"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class Camera(Base):
    """Camera model - represents a traffic camera location"""
    __tablename__ = "cameras"
    
    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    area = Column(String(100), nullable=True)
    
    # Status
    status = Column(String(20), default=CameraStatus.ONLINE)
    is_active = Column(Boolean, default=True)
    
    # Urban Observatory reference
    uo_sensor_id = Column(String(100), nullable=True)  # Urban Observatory sensor ID
    uo_feed_url = Column(String(500), nullable=True)   # Image feed URL
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_image_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="camera", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Camera {self.id}: {self.name}>"


class Detection(Base):
    """Detection model - stores YOLO detection results"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    
    # Detection counts
    vehicles = Column(Integer, default=0)
    pedestrians = Column(Integer, default=0)
    cyclists = Column(Integer, default=0)
    
    # Additional vehicle breakdown
    cars = Column(Integer, default=0)
    buses = Column(Integer, default=0)
    trucks = Column(Integer, default=0)
    motorcycles = Column(Integer, default=0)
    
    # Computed metrics
    traffic_level = Column(String(20), default=TrafficLevel.UNKNOWN)
    congestion_score = Column(Float, default=0.0)  # 0-100 scale
    
    # Detection metadata
    confidence_avg = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    image_path = Column(String(500), nullable=True)
    
    # Timestamps
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera = relationship("Camera", back_populates="detections")
    
    def __repr__(self):
        return f"<Detection {self.id}: Camera {self.camera_id} at {self.detected_at}>"


class HourlyMetric(Base):
    """Hourly aggregated metrics for analytics"""
    __tablename__ = "hourly_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=True)  # Null for city-wide
    
    # Time period
    hour_start = Column(DateTime(timezone=True), nullable=False)
    
    # Aggregated counts
    total_vehicles = Column(Integer, default=0)
    total_pedestrians = Column(Integer, default=0)
    total_cyclists = Column(Integer, default=0)
    
    # Statistics
    avg_vehicles = Column(Float, default=0.0)
    avg_pedestrians = Column(Float, default=0.0)
    avg_cyclists = Column(Float, default=0.0)
    
    max_vehicles = Column(Integer, default=0)
    max_pedestrians = Column(Integer, default=0)
    
    # Traffic analysis
    avg_congestion_score = Column(Float, default=0.0)
    peak_traffic_level = Column(String(20), nullable=True)
    
    # Detection count
    detection_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<HourlyMetric {self.hour_start}: {self.total_vehicles} vehicles>"


class SystemLog(Base):
    """System logs for monitoring and debugging"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    component = Column(String(50), nullable=False)  # detector, collector, api
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SystemLog {self.level}: {self.message[:50]}>"
