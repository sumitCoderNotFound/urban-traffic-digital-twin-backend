"""
Metrics API Router
Endpoints for traffic analytics and aggregated data
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.db.database import get_db
from app.models.models import Camera, Detection, HourlyMetric
from app.schemas.schemas import (
    TotalMetrics, TrafficMetrics, HourlyDataPoint, CameraMetrics, TrafficLevelEnum
)

router = APIRouter()


@router.get("/totals", response_model=TotalMetrics)
async def get_total_metrics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get current total metrics across all cameras.
    
    Returns aggregated vehicle, pedestrian, and cyclist counts
    from the most recent detections.
    """
    # Get all active cameras
    cameras_query = select(Camera).where(Camera.is_active == True)
    cameras_result = await db.execute(cameras_query)
    cameras = cameras_result.scalars().all()
    
    total_vehicles = 0
    total_pedestrians = 0
    total_cyclists = 0
    active_cameras = 0
    
    for camera in cameras:
        # Get latest detection for each camera
        detection_query = (
            select(Detection)
            .where(Detection.camera_id == camera.id)
            .order_by(Detection.detected_at.desc())
            .limit(1)
        )
        detection_result = await db.execute(detection_query)
        detection = detection_result.scalar_one_or_none()
        
        if detection:
            total_vehicles += detection.vehicles
            total_pedestrians += detection.pedestrians
            total_cyclists += detection.cyclists
        
        if camera.status == "online":
            active_cameras += 1
    
    return TotalMetrics(
        vehicles=total_vehicles,
        pedestrians=total_pedestrians,
        cyclists=total_cyclists,
        active_cameras=active_cameras,
        total_cameras=len(cameras),
    )


@router.get("/hourly", response_model=List[HourlyDataPoint])
async def get_hourly_metrics(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to retrieve"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get hourly aggregated traffic data.
    
    Returns data points for charting traffic patterns over time.
    """
    now = datetime.utcnow()
    hourly_data = []
    
    for i in range(hours - 1, -1, -1):
        hour_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=i)
        hour_end = hour_start + timedelta(hours=1)
        
        # Build query
        query = (
            select(
                func.coalesce(func.avg(Detection.vehicles), 0).label("avg_vehicles"),
                func.coalesce(func.avg(Detection.pedestrians), 0).label("avg_pedestrians"),
                func.coalesce(func.avg(Detection.cyclists), 0).label("avg_cyclists"),
            )
            .where(Detection.detected_at >= hour_start)
            .where(Detection.detected_at < hour_end)
        )
        
        if camera_id:
            query = query.where(Detection.camera_id == camera_id)
        
        result = await db.execute(query)
        row = result.one()
        
        hourly_data.append(HourlyDataPoint(
            time=hour_start.strftime("%H:%M"),
            vehicles=int(row.avg_vehicles or 0),
            pedestrians=int(row.avg_pedestrians or 0),
            cyclists=int(row.avg_cyclists or 0),
        ))
    
    return hourly_data


@router.get("/traffic", response_model=TrafficMetrics)
async def get_traffic_metrics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive traffic metrics.
    
    Includes totals, hourly data, peak hour, and average congestion.
    """
    # Get totals
    totals = await get_total_metrics(db)
    
    # Get hourly data
    hourly_data = await get_hourly_metrics(hours=24, db=db)
    
    # Find peak hour
    peak_hour = None
    max_total = 0
    for data_point in hourly_data:
        total = data_point.vehicles + data_point.pedestrians
        if total > max_total:
            max_total = total
            peak_hour = data_point.time
    
    # Calculate average congestion
    now = datetime.utcnow()
    yesterday = now - timedelta(hours=24)
    
    congestion_query = (
        select(func.avg(Detection.congestion_score))
        .where(Detection.detected_at >= yesterday)
    )
    congestion_result = await db.execute(congestion_query)
    avg_congestion = congestion_result.scalar() or 0.0
    
    return TrafficMetrics(
        totals=totals,
        hourly_data=hourly_data,
        peak_hour=peak_hour,
        avg_congestion=round(avg_congestion, 2),
        last_updated=datetime.utcnow(),
    )


@router.get("/camera/{camera_id}", response_model=CameraMetrics)
async def get_camera_metrics(
    camera_id: str,
    hours: int = Query(24, ge=1, le=168),
    db: AsyncSession = Depends(get_db)
):
    """
    Get metrics for a specific camera.
    """
    # Get camera
    camera_query = select(Camera).where(Camera.id == camera_id)
    camera_result = await db.execute(camera_query)
    camera = camera_result.scalar_one_or_none()
    
    if not camera:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Get aggregated metrics
    since = datetime.utcnow() - timedelta(hours=hours)
    
    metrics_query = (
        select(
            func.sum(Detection.vehicles).label("total_vehicles"),
            func.sum(Detection.pedestrians).label("total_pedestrians"),
            func.sum(Detection.cyclists).label("total_cyclists"),
            func.avg(Detection.vehicles).label("avg_vehicles"),
            func.count(Detection.id).label("detection_count"),
        )
        .where(Detection.camera_id == camera_id)
        .where(Detection.detected_at >= since)
    )
    
    result = await db.execute(metrics_query)
    row = result.one()
    
    # Get current traffic level
    latest_query = (
        select(Detection)
        .where(Detection.camera_id == camera_id)
        .order_by(Detection.detected_at.desc())
        .limit(1)
    )
    latest_result = await db.execute(latest_query)
    latest = latest_result.scalar_one_or_none()
    
    # Calculate peak hour
    hourly_data = await get_hourly_metrics(hours=hours, camera_id=camera_id, db=db)
    peak_hour = None
    max_total = 0
    for data_point in hourly_data:
        total = data_point.vehicles + data_point.pedestrians
        if total > max_total:
            max_total = total
            peak_hour = data_point.time
    
    return CameraMetrics(
        camera_id=camera_id,
        camera_name=camera.name,
        total_vehicles=int(row.total_vehicles or 0),
        total_pedestrians=int(row.total_pedestrians or 0),
        total_cyclists=int(row.total_cyclists or 0),
        avg_vehicles_per_hour=round(float(row.avg_vehicles or 0), 2),
        peak_hour=peak_hour,
        current_traffic_level=TrafficLevelEnum(latest.traffic_level) if latest else TrafficLevelEnum.UNKNOWN,
    )


@router.get("/summary")
async def get_metrics_summary(
    db: AsyncSession = Depends(get_db)
):
    """
    Get a quick summary of current system metrics.
    """
    # Camera counts
    cameras_query = select(func.count(Camera.id)).where(Camera.is_active == True)
    cameras_result = await db.execute(cameras_query)
    total_cameras = cameras_result.scalar()
    
    online_query = (
        select(func.count(Camera.id))
        .where(Camera.is_active == True)
        .where(Camera.status == "online")
    )
    online_result = await db.execute(online_query)
    online_cameras = online_result.scalar()
    
    # Detection counts (last 24h)
    yesterday = datetime.utcnow() - timedelta(hours=24)
    detections_query = (
        select(func.count(Detection.id))
        .where(Detection.detected_at >= yesterday)
    )
    detections_result = await db.execute(detections_query)
    detection_count = detections_result.scalar()
    
    # Traffic level distribution
    level_query = (
        select(Detection.traffic_level, func.count(Detection.id))
        .where(Detection.detected_at >= yesterday)
        .group_by(Detection.traffic_level)
    )
    level_result = await db.execute(level_query)
    level_distribution = {row[0]: row[1] for row in level_result.all()}
    
    return {
        "cameras": {
            "total": total_cameras,
            "online": online_cameras,
            "offline": total_cameras - online_cameras,
        },
        "detections_24h": detection_count,
        "traffic_distribution": level_distribution,
        "last_updated": datetime.utcnow().isoformat(),
    }
