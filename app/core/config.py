"""
Application Configuration
Loads settings from environment variables
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Project Info
    PROJECT_NAME: str = "Urban Digital Twin API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/urban_twin"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/urban_twin"
    
    # SQLite fallback for development
    SQLITE_URL: str = "sqlite+aiosqlite:///./urban_twin.db"
    USE_SQLITE: bool = True  # Set to False for PostgreSQL
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Urban Observatory API
    URBAN_OBSERVATORY_BASE_URL: str = "https://api.newcastle.urbanobservatory.ac.uk/api/v2"
    URBAN_OBSERVATORY_TIMEOUT: int = 30
    
    # YOLO Settings
    YOLO_MODEL: str = "yolov8n.pt"  # nano model for speed, use yolov8m.pt for accuracy
    YOLO_CONFIDENCE: float = 0.5
    YOLO_DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # Detection Classes (COCO dataset)
    DETECTION_CLASSES: dict = {
        0: "person",      # pedestrians
        1: "bicycle",     # cyclists
        2: "car",         # vehicles
        3: "motorcycle",  # vehicles
        5: "bus",         # vehicles
        7: "truck",       # vehicles
    }
    
    # Data Collection
    COLLECTION_INTERVAL: int = 300  # 5 minutes in seconds
    IMAGE_STORAGE_PATH: str = "./data/images"
    
    # Traffic Thresholds
    TRAFFIC_HIGH_VEHICLES: int = 40
    TRAFFIC_HIGH_PEDESTRIANS: int = 150
    TRAFFIC_MEDIUM_VEHICLES: int = 20
    TRAFFIC_MEDIUM_PEDESTRIANS: int = 80
    
    @property
    def database_url(self) -> str:
        """Return appropriate database URL"""
        if self.USE_SQLITE:
            return self.SQLITE_URL
        return self.DATABASE_URL
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
