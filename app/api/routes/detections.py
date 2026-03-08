"""
Detections API Router
Endpoints for managing YOLO detection records and running
the live detection pipeline against Newcastle Urban Observatory cameras.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
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


# ── Read endpoints ────────────────────────────────────────────────────────────

@router.get("/", response_model=List[DetectionResponse])
async def get_detections(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours of history to return"),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """Return detection records, optionally filtered by camera and time range."""
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
async def get_latest_detections(db: AsyncSession = Depends(get_db)):
    """Return the most recent detection for each active camera."""
    cameras_result = await db.execute(
        select(Camera).where(Camera.is_active == True)
    )
    cameras = cameras_result.scalars().all()

    latest = []
    for camera in cameras:
        result = await db.execute(
            select(Detection)
            .where(Detection.camera_id == camera.id)
            .order_by(Detection.detected_at.desc())
            .limit(1)
        )
        detection = result.scalar_one_or_none()
        if detection:
            latest.append(
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

    return latest


@router.get("/{detection_id}", response_model=DetectionDetailResponse)
async def get_detection(
    detection_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Return detailed information for a single detection record."""
    result = await db.execute(
        select(Detection).where(Detection.id == detection_id)
    )
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


# ── Write endpoints ───────────────────────────────────────────────────────────

@router.post("/run")
async def run_detection_pipeline(
    max_cameras: int = Query(10, ge=1, le=50, description="Number of cameras to process"),
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch live images from Newcastle Urban Observatory cameras and
    run YOLOv8 object detection on each one.

    Saves a Camera record and a Detection record for every camera
    processed, then returns a summary of what was found.
    """
    from app.services.urban_observatory import UrbanObservatoryCollector
    from app.services.yolo_detector import get_detector
    from app.services.traffic_calculator import calculate_traffic_level, calculate_congestion_score

    collector = UrbanObservatoryCollector()
    detector = get_detector()

    # Fetch camera list from Urban Observatory
    cameras = await collector.fetch_cameras_from_api()
    if not cameras:
        raise HTTPException(
            status_code=503,
            detail="Could not reach Newcastle Urban Observatory API"
        )

    cameras = cameras[:max_cameras]
    processed = 0
    saved = 0

    for cam in cameras:
        # Download the camera image
        image_path = await collector.fetch_camera_image(
            cam.get("image_url", ""),
            cam.get("camera_id", "unknown")
        )
        if not image_path:
            continue

        processed += 1

        # Run YOLOv8 detection
        detection = detector.detect(image_path)
        if not detection:
            continue

        camera_id = cam["camera_id"]
        camera_name = cam.get("name", camera_id)

        # Upsert camera record
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        db_camera = result.scalar_one_or_none()

        if db_camera:
            db_camera.latitude = cam.get("latitude", 54.9783)
            db_camera.longitude = cam.get("longitude", -1.6178)
            db_camera.status = "online"
            db_camera.last_image_at = datetime.utcnow()
        else:
            db_camera = Camera(
                id=camera_id,
                name=camera_name,
                latitude=cam.get("latitude", 54.9783),
                longitude=cam.get("longitude", -1.6178),
                area=cam.get("place", "Newcastle"),
                status="online",
                is_active=True,
                last_image_at=datetime.utcnow(),
            )
            db.add(db_camera)

        # Save detection result
        traffic_level = calculate_traffic_level(detection.vehicles, detection.pedestrians)
        congestion_score = calculate_congestion_score(
            detection.vehicles, detection.pedestrians, detection.cyclists
        )

        db_detection = Detection(
            camera_id=camera_id,
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
            image_path=image_path,
        )
        db.add(db_detection)

        await db.commit()
        saved += 1

    return {
        "status": "complete",
        "cameras_requested": max_cameras,
        "cameras_processed": processed,
        "results_saved": saved,
        "message": f"Detection complete — {processed} cameras processed, {saved} results saved to database"
    }


@router.post("/", response_model=DetectionResponse, status_code=201)
async def create_detection(
    detection: DetectionCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a single detection record manually."""
    camera_result = await db.execute(
        select(Camera).where(Camera.id == detection.camera_id)
    )
    camera = camera_result.scalar_one_or_none()

    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {detection.camera_id} not found")

    traffic_level = calculate_traffic_level(detection.vehicles, detection.pedestrians)
    congestion_score = calculate_congestion_score(
        detection.vehicles, detection.pedestrians, detection.cyclists
    )

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
    """Create multiple detection records in a single request."""
    created = []

    for detection in detections:
        camera_result = await db.execute(
            select(Camera).where(Camera.id == detection.camera_id)
        )
        camera = camera_result.scalar_one_or_none()

        if not camera:
            continue

        traffic_level = calculate_traffic_level(detection.vehicles, detection.pedestrians)
        congestion_score = calculate_congestion_score(
            detection.vehicles, detection.pedestrians, detection.cyclists
        )

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
        camera.last_image_at = datetime.utcnow()
        created.append(db_detection)

    await db.commit()
    for d in created:
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
        for d in created
    ]


@router.delete("/{detection_id}", status_code=204)
async def delete_detection(
    detection_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a detection record."""
    result = await db.execute(
        select(Detection).where(Detection.id == detection_id)
    )
    detection = result.scalar_one_or_none()

    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")

    await db.delete(detection)
    await db.commit()