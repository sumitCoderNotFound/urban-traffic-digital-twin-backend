import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import async_session_maker
from app.models.models import Camera, Detection
from app.services.urban_observatory import UrbanObservatoryCollector
from app.services.yolo_detector import get_detector
from app.services.traffic_calculator import calculate_traffic_level, calculate_congestion_score
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# How many cameras to process per run
# 20 gives good coverage without taking too long (~3-4 mins per run on CPU)
CAMERAS_PER_RUN = 20

# How often to run in seconds (15 minutes = 900)
POLL_INTERVAL_SECONDS = 900

# Where to save images for the training dataset
DATASET_PATH = Path("./data/dataset")


class DetectionScheduler:
    """
    Runs automatic detection on a fixed interval.

    Each run:
    - Fetches CAMERAS_PER_RUN cameras from Urban Observatory
    - Rotates through cameras so all 345 get covered over time
    - Saves images to data/dataset/ with descriptive filenames
    - Runs YOLOv8 and saves detections to DB
    """

    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._run_count = 0
        self._camera_offset = 0  # rotates each run to cover all cameras
        DATASET_PATH.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the background scheduler loop."""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Scheduler started — running every {POLL_INTERVAL_SECONDS // 60} minutes")
        print(f"\n⏰ Auto-scheduler started — detection runs every {POLL_INTERVAL_SECONDS // 60} mins\n")

    async def stop(self):
        """Stop the scheduler gracefully."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("\n🛑 Scheduler stopped")

    async def _loop(self):
        """Main scheduler loop — runs immediately then waits POLL_INTERVAL_SECONDS."""
        while self.running:
            try:
                await self._run_collection()
            except Exception as e:
                logger.error(f"Scheduler run failed: {e}")
                print(f"\n❌ Scheduler error: {e}")

            if self.running:
                next_run = datetime.utcnow().strftime("%H:%M:%S")
                print(f"\n⏳ Next detection run in {POLL_INTERVAL_SECONDS // 60} mins "
                      f"(approx {POLL_INTERVAL_SECONDS // 60} mins from now)")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _run_collection(self):
        """
        Single collection cycle:
        1. Fetch camera list from Urban Observatory
        2. Select a rotating slice of CAMERAS_PER_RUN cameras
        3. Download image for each camera
        4. Save image to dataset folder
        5. Run YOLOv8 detection
        6. Save results to database
        """
        self._run_count += 1
        timestamp = datetime.utcnow()

        print(f"\n{'='*60}")
        print(f"📸 AUTO-DETECTION RUN #{self._run_count}")
        print(f"   Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'='*60}")

        collector = UrbanObservatoryCollector()
        detector = get_detector()

        # Fetch full camera list
        all_cameras = await collector.fetch_cameras_from_api()
        if not all_cameras:
            print("   ⚠️  No cameras returned from Urban Observatory")
            return

        total = len(all_cameras)
        print(f"   📡 {total} cameras available")

        # Rotate through cameras across runs
        # Run 1: cameras 0-19, Run 2: cameras 20-39, etc.
        start = self._camera_offset % total
        selected = []
        for i in range(CAMERAS_PER_RUN):
            selected.append(all_cameras[(start + i) % total])

        self._camera_offset += CAMERAS_PER_RUN
        print(f"   🎯 Processing cameras {start}–{(start + CAMERAS_PER_RUN - 1) % total} "
              f"(offset {self._camera_offset}/{total})\n")

        processed = 0
        saved = 0

        async with async_session_maker() as db:
            for cam in selected:
                image_url = cam.get("image_url", "")
                camera_id = cam.get("camera_id", "unknown")
                camera_name = cam.get("name", camera_id)

                # Download image
                image_path = await collector.fetch_camera_image(image_url, camera_id)
                if not image_path:
                    continue

                processed += 1

                # Also save a copy to the dataset folder with a descriptive name
                await self._save_to_dataset(image_path, camera_name, timestamp)

                # Run YOLO detection
                detection = detector.detect(image_path)
                if not detection:
                    continue

                # Upsert camera record
                result = await db.execute(select(Camera).where(Camera.id == camera_id))
                db_camera = result.scalar_one_or_none()

                if db_camera:
                    db_camera.latitude = cam.get("latitude", 54.9783)
                    db_camera.longitude = cam.get("longitude", -1.6178)
                    db_camera.status = "online"
                    db_camera.last_image_at = timestamp
                else:
                    db_camera = Camera(
                        id=camera_id,
                        name=camera_name,
                        latitude=cam.get("latitude", 54.9783),
                        longitude=cam.get("longitude", -1.6178),
                        area=cam.get("place", "Newcastle"),
                        status="online",
                        is_active=True,
                        last_image_at=timestamp,
                    )
                    db.add(db_camera)

                # Save detection
                traffic_level = calculate_traffic_level(detection.vehicles, detection.pedestrians)
                congestion_score = calculate_congestion_score(
                    detection.vehicles, detection.pedestrians, detection.cyclists
                )

                db.add(Detection(
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
                ))

                await db.commit()
                saved += 1

                print(f"   ✅ {camera_name[:40]:<40} "
                      f"V:{detection.vehicles:>3} P:{detection.pedestrians:>3} C:{detection.cyclists:>2}")

        print(f"\n   Run #{self._run_count} complete — "
              f"{processed} images downloaded, {saved} detections saved")
        print(f"   Dataset total: {len(list(DATASET_PATH.glob('*.jpg')))} images")

    async def _save_to_dataset(self, image_path: str, camera_name: str, timestamp: datetime):
        """
        Copy image to dataset folder with a clean descriptive filename.

        Filename format: A167_Darlington_Rd_20260309_134500.jpg
        These are used for model training — Roboflow can import this folder directly.
        """
        import shutil
        import re

        # Clean camera name for filename
        clean_name = re.sub(r"[^\w\s-]", "", camera_name)
        clean_name = re.sub(r"\s+", "_", clean_name.strip())[:40]
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{clean_name}_{time_str}.jpg"

        dest = DATASET_PATH / filename
        try:
            shutil.copy2(image_path, dest)
        except Exception as e:
            logger.debug(f"Dataset copy failed for {camera_name}: {e}")

    @property
    def status(self) -> dict:
        """Return current scheduler status."""
        return {
            "running": self.running,
            "run_count": self._run_count,
            "cameras_per_run": CAMERAS_PER_RUN,
            "interval_minutes": POLL_INTERVAL_SECONDS // 60,
            "dataset_images": len(list(DATASET_PATH.glob("*.jpg"))) if DATASET_PATH.exists() else 0,
        }


# Single instance used across the app
scheduler = DetectionScheduler()