"""
Health Check API Router
Endpoints for system health monitoring
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime

from app.db.database import get_db
from app.core.config import settings
from app.schemas.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db)
):
    """
    Check system health status.
    
    Returns status of API, database connection, and detector service.
    """
    # Check database connection
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check detector availability
    detector_status = "ready"
    try:
        from app.services.yolo_detector import YOLODetector
        detector = YOLODetector()
        if not detector.is_available():
            detector_status = "model not loaded"
    except ImportError:
        detector_status = "not configured"
    except Exception as e:
        detector_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        version=settings.VERSION,
        database=db_status,
        detector=detector_status,
        timestamp=datetime.utcnow(),
    )


@router.get("/health/live")
async def liveness_check():
    """
    Simple liveness check for Kubernetes/container orchestration.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db)
):
    """
    Readiness check to verify the service can handle requests.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Database not ready: {str(e)}")
