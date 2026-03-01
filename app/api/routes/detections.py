"""
Detections API Router
Endpoints for managing detection records
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.models import Detection, Camera
from app.schemas.schemas import (
    DetectionCreate, DetectionResponse, DetectionDetailResponse, TrafficLevelEnum
)
from app.services.traffic_calculator import calculate_traffic_level, calculate_congestion_score

router = APIRouter()


@router.get("/", response_model=List[DetectionResponse])
async def get_detections(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours of history"),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detection records with optional filters.
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    query = select(Detection).where(Detection.detected_at >= since)
    
    if camera_id:
        query = query.where(Detection.camera_id == camera_id)
    
    query = query.order_by(Detection.detected_at.desc()).limit(limit)
    
    result = await db.execute(query)
    detections = result.scalars().all()
    
    return [
        DetectionResponse(
            id=d.id,
            camera_id=d.camera_id,
            vehicles=d.vehicles,
            pedestrians=d.pedestrians,
            cyclists=d.cyclists,
            traffic_level=d.traffic_level,
            congestion_score=d.congestion_score,
            detected_at=d.detected_at,
        )
        for d in detections
    ]


@router.get("/latest", response_model=List[DetectionResponse])
async def get_latest_detections(
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most recent detection for each camera.
    """
    # Get all active cameras
    cameras_query = select(Camera).where(Camera.is_active == True)
    cameras_result = await db.execute(cameras_query)
    cameras = cameras_result.scalars().all()
    
    latest_detections = []
    for camera in cameras:
        query = (
            select(Detection)
            .where(Detection.camera_id == camera.id)
            .order_by(Detection.detected_at.desc())
            .limit(1)
        )
        result = await db.execute(query)
        detection = result.scalar_one_or_none()
        
        if detection:
            latest_detections.append(
                DetectionResponse(
                    id=detection.id,
                    camera_id=detection.camera_id,
                    vehicles=detection.vehicles,
                    pedestrians=detection.pedestrians,
                    cyclists=detection.cyclists,
                    traffic_level=detection.traffic_level,
                    congestion_score=detection.congestion_score,
                    detected_at=detection.detected_at,
                )
            )
    
    return latest_detections


@router.get("/{detection_id}", response_model=DetectionDetailResponse)
async def get_detection(
    detection_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific detection.
    """
    query = select(Detection).where(Detection.id == detection_id)
    result = await db.execute(query)
    detection = result.scalar_one_or_none()
    
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    return DetectionDetailResponse(
        id=detection.id,
        camera_id=detection.camera_id,
        vehicles=detection.vehicles,
        pedestrians=detection.pedestrians,
        cyclists=detection.cyclists,
        traffic_level=detection.traffic_level,
        congestion_score=detection.congestion_score,
        detected_at=detection.detected_at,
        cars=detection.cars,
        buses=detection.buses,
        trucks=detection.trucks,
        motorcycles=detection.motorcycles,
        confidence_avg=detection.confidence_avg,
        processing_time_ms=detection.processing_time_ms,
    )


@router.post("/", response_model=DetectionResponse, status_code=201)
async def create_detection(
    detection: DetectionCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new detection record.
    
    This endpoint is typically called by the YOLO detection service
    after processing a camera image.
    """
    # Verify camera exists
    camera_query = select(Camera).where(Camera.id == detection.camera_id)
    camera_result = await db.execute(camera_query)
    camera = camera_result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {detection.camera_id} not found")
    
    # Calculate traffic level and congestion score
    traffic_level = calculate_traffic_level(detection.vehicles, detection.pedestrians)
    congestion_score = calculate_congestion_score(detection.vehicles, detection.pedestrians, detection.cyclists)
    
    # Create detection record
    db_detection = Detection(
        camera_id=detection.camera_id,
        vehicles=detection.vehicles,
        pedestrians=detection.pedestrians,
        cyclists=detection.cyclists,
        cars=detection.cars,
        buses=detection.buses,
        trucks=detection.trucks,
        motorcycles=detection.motorcycles,
        traffic_level=traffic_level,
        congestion_score=congestion_score,
        confidence_avg=detection.confidence_avg,
        processing_time_ms=detection.processing_time_ms,
        image_path=detection.image_path,
    )
    
    db.add(db_detection)
    
    # Update camera's last image timestamp
    camera.last_image_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_detection)
    
    return DetectionResponse(
        id=db_detection.id,
        camera_id=db_detection.camera_id,
        vehicles=db_detection.vehicles,
        pedestrians=db_detection.pedestrians,
        cyclists=db_detection.cyclists,
        traffic_level=db_detection.traffic_level,
        congestion_score=db_detection.congestion_score,
        detected_at=db_detection.detected_at,
    )


@router.post("/batch", response_model=List[DetectionResponse], status_code=201)
async def create_detections_batch(
    detections: List[DetectionCreate],
    db: AsyncSession = Depends(get_db)
):
    """
    Create multiple detection records in a single request.
    
    Useful for batch processing of images.
    """
    created_detections = []
    
    for detection in detections:
        # Verify camera exists
        camera_query = select(Camera).where(Camera.id == detection.camera_id)
        camera_result = await db.execute(camera_query)
        camera = camera_result.scalar_one_or_none()
        
        if not camera:
            continue  # Skip invalid cameras
        
        # Calculate metrics
        traffic_level = calculate_traffic_level(detection.vehicles, detection.pedestrians)
        congestion_score = calculate_congestion_score(detection.vehicles, detection.pedestrians, detection.cyclists)
        
        # Create record
        db_detection = Detection(
            camera_id=detection.camera_id,
            vehicles=detection.vehicles,
            pedestrians=detection.pedestrians,
            cyclists=detection.cyclists,
            cars=detection.cars,
            buses=detection.buses,
            trucks=detection.trucks,
            motorcycles=detection.motorcycles,
            traffic_level=traffic_level,
            congestion_score=congestion_score,
            confidence_avg=detection.confidence_avg,
            processing_time_ms=detection.processing_time_ms,
            image_path=detection.image_path,
        )
        
        db.add(db_detection)
        created_detections.append(db_detection)
        
        # Update camera timestamp
        camera.last_image_at = datetime.utcnow()
    
    await db.commit()
    
    # Refresh all detections
    for d in created_detections:
        await db.refresh(d)
    
    return [
        DetectionResponse(
            id=d.id,
            camera_id=d.camera_id,
            vehicles=d.vehicles,
            pedestrians=d.pedestrians,
            cyclists=d.cyclists,
            traffic_level=d.traffic_level,
            congestion_score=d.congestion_score,
            detected_at=d.detected_at,
        )
        for d in created_detections
    ]


@router.delete("/{detection_id}", status_code=204)
async def delete_detection(
    detection_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a detection record.
    """
    query = select(Detection).where(Detection.id == detection_id)
    result = await db.execute(query)
    detection = result.scalar_one_or_none()
    
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    await db.delete(detection)
    await db.commit()
    
    return None
