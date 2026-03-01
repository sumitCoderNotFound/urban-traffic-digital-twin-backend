"""
Cameras API Router
Endpoints for camera management and data retrieval
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.models import Camera, Detection, TrafficLevel
from app.schemas.schemas import (
    CameraResponse, CameraDetailResponse, CameraCreate, CameraUpdate,
    Location, DetectionResponse, TrafficLevelEnum
)
from app.services.traffic_calculator import calculate_traffic_level

router = APIRouter()


@router.get("/", response_model=List[CameraResponse])
async def get_all_cameras(
    status: Optional[str] = Query(None, description="Filter by status (online/offline)"),
    area: Optional[str] = Query(None, description="Filter by area"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all cameras with their current detection data.
    
    Returns a list of cameras with latest vehicle, pedestrian, and cyclist counts.
    """
    query = select(Camera).where(Camera.is_active == True)
    
    if status:
        query = query.where(Camera.status == status)
    if area:
        query = query.where(Camera.area == area)
    
    result = await db.execute(query)
    cameras = result.scalars().all()
    
    # Get latest detection for each camera
    camera_responses = []
    for camera in cameras:
        # Get latest detection
        detection_query = (
            select(Detection)
            .where(Detection.camera_id == camera.id)
            .order_by(Detection.detected_at.desc())
            .limit(1)
        )
        detection_result = await db.execute(detection_query)
        latest_detection = detection_result.scalar_one_or_none()
        
        camera_data = CameraResponse(
            id=camera.id,
            name=camera.name,
            description=camera.description,
            latitude=camera.latitude,
            longitude=camera.longitude,
            area=camera.area,
            location=Location(lat=camera.latitude, lng=camera.longitude),
            status=camera.status,
            traffic_level=latest_detection.traffic_level if latest_detection else TrafficLevelEnum.UNKNOWN,
            vehicles=latest_detection.vehicles if latest_detection else 0,
            pedestrians=latest_detection.pedestrians if latest_detection else 0,
            cyclists=latest_detection.cyclists if latest_detection else 0,
            last_update=latest_detection.detected_at if latest_detection else camera.updated_at,
            is_active=camera.is_active,
        )
        camera_responses.append(camera_data)
    
    return camera_responses


@router.get("/{camera_id}", response_model=CameraDetailResponse)
async def get_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information for a specific camera.
    
    Includes recent detection history.
    """
    query = select(Camera).where(Camera.id == camera_id)
    result = await db.execute(query)
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Get recent detections (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(hours=24)
    detections_query = (
        select(Detection)
        .where(Detection.camera_id == camera_id)
        .where(Detection.detected_at >= yesterday)
        .order_by(Detection.detected_at.desc())
        .limit(100)
    )
    detections_result = await db.execute(detections_query)
    recent_detections = detections_result.scalars().all()
    
    # Get latest for current stats
    latest = recent_detections[0] if recent_detections else None
    
    return CameraDetailResponse(
        id=camera.id,
        name=camera.name,
        description=camera.description,
        latitude=camera.latitude,
        longitude=camera.longitude,
        area=camera.area,
        location=Location(lat=camera.latitude, lng=camera.longitude),
        status=camera.status,
        traffic_level=latest.traffic_level if latest else TrafficLevelEnum.UNKNOWN,
        vehicles=latest.vehicles if latest else 0,
        pedestrians=latest.pedestrians if latest else 0,
        cyclists=latest.cyclists if latest else 0,
        last_update=latest.detected_at if latest else camera.updated_at,
        is_active=camera.is_active,
        uo_sensor_id=camera.uo_sensor_id,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
        recent_detections=[
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
            for d in recent_detections[:20]  # Last 20 detections
        ]
    )


@router.post("/", response_model=CameraResponse, status_code=201)
async def create_camera(
    camera: CameraCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new camera.
    """
    # Check if camera already exists
    existing = await db.execute(select(Camera).where(Camera.id == camera.id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Camera {camera.id} already exists")
    
    db_camera = Camera(
        id=camera.id,
        name=camera.name,
        description=camera.description,
        latitude=camera.latitude,
        longitude=camera.longitude,
        area=camera.area,
        uo_sensor_id=camera.uo_sensor_id,
        uo_feed_url=camera.uo_feed_url,
    )
    
    db.add(db_camera)
    await db.commit()
    await db.refresh(db_camera)
    
    return CameraResponse(
        id=db_camera.id,
        name=db_camera.name,
        description=db_camera.description,
        latitude=db_camera.latitude,
        longitude=db_camera.longitude,
        area=db_camera.area,
        location=Location(lat=db_camera.latitude, lng=db_camera.longitude),
        status=db_camera.status,
        traffic_level=TrafficLevelEnum.UNKNOWN,
        vehicles=0,
        pedestrians=0,
        cyclists=0,
        last_update=db_camera.created_at,
        is_active=db_camera.is_active,
    )


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    camera_update: CameraUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a camera's information.
    """
    query = select(Camera).where(Camera.id == camera_id)
    result = await db.execute(query)
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Update fields
    update_data = camera_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(camera, field, value)
    
    await db.commit()
    await db.refresh(camera)
    
    return CameraResponse(
        id=camera.id,
        name=camera.name,
        description=camera.description,
        latitude=camera.latitude,
        longitude=camera.longitude,
        area=camera.area,
        location=Location(lat=camera.latitude, lng=camera.longitude),
        status=camera.status,
        traffic_level=TrafficLevelEnum.UNKNOWN,
        vehicles=0,
        pedestrians=0,
        cyclists=0,
        last_update=camera.updated_at,
        is_active=camera.is_active,
    )


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a camera (soft delete by setting is_active=False).
    """
    query = select(Camera).where(Camera.id == camera_id)
    result = await db.execute(query)
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    camera.is_active = False
    await db.commit()
    
    return None


@router.get("/{camera_id}/detections", response_model=List[DetectionResponse])
async def get_camera_detections(
    camera_id: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detection history for a specific camera.
    """
    # Verify camera exists
    camera_query = select(Camera).where(Camera.id == camera_id)
    camera_result = await db.execute(camera_query)
    if not camera_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Get detections
    since = datetime.utcnow() - timedelta(hours=hours)
    query = (
        select(Detection)
        .where(Detection.camera_id == camera_id)
        .where(Detection.detected_at >= since)
        .order_by(Detection.detected_at.desc())
        .limit(limit)
    )
    
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
