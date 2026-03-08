from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):

    PROJECT_NAME: str = "Urban Digital Twin API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    SQLITE_URL: str = "sqlite+aiosqlite:///./urban_twin.db"
    USE_SQLITE: bool = True
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/urban_twin"

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    URBAN_OBSERVATORY_URL: str = "https://portal.cctv.urbanobservatory.ac.uk/latest"
    URBAN_OBSERVATORY_IMAGE_URL: str = "https://portal.cctv.urbanobservatory.ac.uk/photo"
    URBAN_OBSERVATORY_TIMEOUT: int = 30

    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.25
    YOLO_DEVICE: str = "cpu"

    DETECTION_CLASSES: dict = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    COLLECTION_INTERVAL: int = 300
    IMAGE_STORAGE_PATH: str = "./data/images"

    TRAFFIC_HIGH_VEHICLES: int = 40
    TRAFFIC_HIGH_PEDESTRIANS: int = 150
    TRAFFIC_MEDIUM_VEHICLES: int = 20
    TRAFFIC_MEDIUM_PEDESTRIANS: int = 80

    @property
    def database_url(self) -> str:
        if self.USE_SQLITE:
            return self.SQLITE_URL
        return self.DATABASE_URL

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()