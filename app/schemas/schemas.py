"""
Pydantic Schemas
Request and response models for API validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ============================================
# Enums
# ============================================

class CameraStatusEnum(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TrafficLevelEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


# ============================================
# Location Schemas
# ============================================

class Location(BaseModel):
    """Geographic location"""
    lat: float = Field(..., description="Latitude", ge=-90, le=90)
    lng: float = Field(..., description="Longitude", ge=-180, le=180)


# ============================================
# Camera Schemas
# ============================================

class CameraBase(BaseModel):
    """Base camera schema"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    area: Optional[str] = None


class CameraCreate(CameraBase):
    """Schema for creating a new camera"""
    id: str = Field(..., min_length=1, max_length=50)
    uo_sensor_id: Optional[str] = None
    uo_feed_url: Optional[str] = None


class CameraUpdate(BaseModel):
    """Schema for updating a camera"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[CameraStatusEnum] = None
    is_active: Optional[bool] = None


class CameraResponse(CameraBase):
    """Camera response with current detection data"""
    id: str
    location: Location
    status: CameraStatusEnum
    traffic_level: TrafficLevelEnum = TrafficLevelEnum.UNKNOWN
    vehicles: int = 0
    pedestrians: int = 0
    cyclists: int = 0
    last_update: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        from_attributes = True


class CameraDetailResponse(CameraResponse):
    """Detailed camera response with additional info"""
    uo_sensor_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    recent_detections: List["DetectionResponse"] = []


# ============================================
# Detection Schemas
# ============================================

class DetectionBase(BaseModel):
    """Base detection schema"""
    vehicles: int = Field(0, ge=0)
    pedestrians: int = Field(0, ge=0)
    cyclists: int = Field(0, ge=0)


class DetectionCreate(DetectionBase):
    """Schema for creating a detection record"""
    camera_id: str
    cars: int = Field(0, ge=0)
    buses: int = Field(0, ge=0)
    trucks: int = Field(0, ge=0)
    motorcycles: int = Field(0, ge=0)
    confidence_avg: Optional[float] = None
    processing_time_ms: Optional[int] = None
    image_path: Optional[str] = None


class DetectionResponse(DetectionBase):
    """Detection response schema"""
    id: int
    camera_id: str
    traffic_level: TrafficLevelEnum
    congestion_score: float
    detected_at: datetime
    
    class Config:
        from_attributes = True


class DetectionDetailResponse(DetectionResponse):
    """Detailed detection with vehicle breakdown"""
    cars: int = 0
    buses: int = 0
    trucks: int = 0
    motorcycles: int = 0
    confidence_avg: Optional[float] = None
    processing_time_ms: Optional[int] = None


# ============================================
# Metrics Schemas
# ============================================

class TotalMetrics(BaseModel):
    """Total metrics across all cameras"""
    vehicles: int
    pedestrians: int
    cyclists: int
    active_cameras: int
    total_cameras: int


class HourlyDataPoint(BaseModel):
    """Single data point for hourly chart"""
    time: str
    vehicles: int
    pedestrians: int
    cyclists: int


class TrafficMetrics(BaseModel):
    """Traffic metrics response"""
    totals: TotalMetrics
    hourly_data: List[HourlyDataPoint]
    peak_hour: Optional[str] = None
    avg_congestion: float = 0.0
    last_updated: datetime


class CameraMetrics(BaseModel):
    """Metrics for a specific camera"""
    camera_id: str
    camera_name: str
    total_vehicles: int
    total_pedestrians: int
    total_cyclists: int
    avg_vehicles_per_hour: float
    peak_hour: Optional[str] = None
    current_traffic_level: TrafficLevelEnum


# ============================================
# API Response Schemas
# ============================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    detector: str
    timestamp: datetime


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List
    total: int
    page: int
    per_page: int
    pages: int


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# Urban Observatory Schemas
# ============================================

class UOCameraInfo(BaseModel):
    """Urban Observatory camera information"""
    sensor_id: str
    name: str
    latitude: float
    longitude: float
    feed_url: Optional[str] = None
    last_reading: Optional[datetime] = None


# Update forward references
CameraDetailResponse.model_rebuild()
