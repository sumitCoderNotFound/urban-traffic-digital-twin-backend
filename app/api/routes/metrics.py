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
async def get_total_metrics(db: AsyncSession = Depends(get_db)):
    """
    Get current total metrics across all cameras.
    Uses the most recent detection per camera.
    """
    cameras_result = await db.execute(
        select(Camera).where(Camera.is_active == True)
    )
    cameras = cameras_result.scalars().all()

    total_vehicles = 0
    total_pedestrians = 0
    total_cyclists = 0
    active_cameras = 0

    for camera in cameras:
        detection_result = await db.execute(
            select(Detection)
            .where(Detection.camera_id == camera.id)
            .order_by(Detection.detected_at.desc())
            .limit(1)
        )
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
    Get hourly aggregated traffic data for charting.

    For each hour, takes the LATEST detection per camera then averages
    across all cameras. This means repeated Run Detection clicks in the
    same hour do not inflate the chart — each camera contributes exactly
    one reading per hour regardless of how many times detection ran.
    """
    now = datetime.utcnow()
    hourly_data = []

    # Get distinct camera IDs
    if camera_id:
        camera_ids = [camera_id]
    else:
        cam_result = await db.execute(select(Detection.camera_id).distinct())
        camera_ids = [row[0] for row in cam_result.all()]

    for i in range(hours - 1, -1, -1):
        hour_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=i)
        hour_end = hour_start + timedelta(hours=1)

        hour_vehicles, hour_pedestrians, hour_cyclists = [], [], []

        for cam_id in camera_ids:
            result = await db.execute(
                select(Detection)
                .where(Detection.camera_id == cam_id)
                .where(Detection.detected_at >= hour_start)
                .where(Detection.detected_at < hour_end)
                .order_by(Detection.detected_at.desc())
                .limit(1)
            )
            det = result.scalar_one_or_none()
            if det:
                hour_vehicles.append(det.vehicles)
                hour_pedestrians.append(det.pedestrians)
                hour_cyclists.append(det.cyclists)

        def safe_avg(lst):
            return int(sum(lst) / len(lst)) if lst else 0

        hourly_data.append(HourlyDataPoint(
            time=hour_start.strftime("%H:%M"),
            vehicles=safe_avg(hour_vehicles),
            pedestrians=safe_avg(hour_pedestrians),
            cyclists=safe_avg(hour_cyclists),
        ))

    return hourly_data


@router.get("/traffic", response_model=TrafficMetrics)
async def get_traffic_metrics(db: AsyncSession = Depends(get_db)):
    """Get comprehensive traffic metrics including totals, hourly data and peak hour."""
    totals = await get_total_metrics(db)
    hourly_data = await get_hourly_metrics(hours=24, db=db)

    peak_hour = None
    max_total = 0
    for dp in hourly_data:
        total = dp.vehicles + dp.pedestrians
        if total > max_total:
            max_total = total
            peak_hour = dp.time

    congestion_result = await db.execute(
        select(func.avg(Detection.congestion_score))
        .where(Detection.detected_at >= datetime.utcnow() - timedelta(hours=24))
    )
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
    """Get metrics for a specific camera."""
    camera_result = await db.execute(
        select(Camera).where(Camera.id == camera_id)
    )
    camera = camera_result.scalar_one_or_none()

    if not camera:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    since = datetime.utcnow() - timedelta(hours=hours)

    result = await db.execute(
        select(
            func.sum(Detection.vehicles).label("total_vehicles"),
            func.sum(Detection.pedestrians).label("total_pedestrians"),
            func.sum(Detection.cyclists).label("total_cyclists"),
            func.avg(Detection.vehicles).label("avg_vehicles"),
        )
        .where(Detection.camera_id == camera_id)
        .where(Detection.detected_at >= since)
    )
    row = result.one()

    latest_result = await db.execute(
        select(Detection)
        .where(Detection.camera_id == camera_id)
        .order_by(Detection.detected_at.desc())
        .limit(1)
    )
    latest = latest_result.scalar_one_or_none()

    hourly_data = await get_hourly_metrics(hours=hours, camera_id=camera_id, db=db)
    peak_hour = None
    max_total = 0
    for dp in hourly_data:
        total = dp.vehicles + dp.pedestrians
        if total > max_total:
            max_total = total
            peak_hour = dp.time

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
async def get_metrics_summary(db: AsyncSession = Depends(get_db)):
    """Quick summary of current system state including last detection time."""
    cameras_result = await db.execute(
        select(func.count(Camera.id)).where(Camera.is_active == True)
    )
    total_cameras = cameras_result.scalar()

    online_result = await db.execute(
        select(func.count(Camera.id))
        .where(Camera.is_active == True)
        .where(Camera.status == "online")
    )
    online_cameras = online_result.scalar()

    yesterday = datetime.utcnow() - timedelta(hours=24)

    detection_count_result = await db.execute(
        select(func.count(Detection.id))
        .where(Detection.detected_at >= yesterday)
    )
    detection_count = detection_count_result.scalar()

    # Last detection run timestamp
    last_detection_result = await db.execute(
        select(Detection.detected_at)
        .order_by(Detection.detected_at.desc())
        .limit(1)
    )
    last_detection_row = last_detection_result.scalar_one_or_none()

    level_result = await db.execute(
        select(Detection.traffic_level, func.count(Detection.id))
        .where(Detection.detected_at >= yesterday)
        .group_by(Detection.traffic_level)
    )
    level_distribution = {row[0]: row[1] for row in level_result.all()}

    return {
        "cameras": {
            "total": total_cameras,
            "online": online_cameras,
            "offline": total_cameras - online_cameras,
        },
        "detections_24h": detection_count,
        "last_detection_at": last_detection_row.isoformat() if last_detection_row else None,
        "traffic_distribution": level_distribution,
        "last_updated": datetime.utcnow().isoformat(),
    }