"""
Remove Mock Seed Data
======================
Deletes the fake cam-001 to cam-010 cameras and all their
associated detections that were inserted by seed_database.py.

These are not real cameras — they were placeholder data used
during early development before the Urban Observatory API
integration was complete.

Run from the project root:
    python scripts/remove_mock_data.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete
from app.db.database import async_session_maker, engine, Base
from app.models.models import Camera, Detection

# IDs inserted by seed_database.py
MOCK_CAMERA_IDS = [
    "cam-001", "cam-002", "cam-003", "cam-004", "cam-005",
    "cam-006", "cam-007", "cam-008", "cam-009", "cam-010",
]


async def remove_mock_data():
    print("\n🧹 Removing mock seed data from database...")
    print("=" * 55)

    async with async_session_maker() as session:

        removed_cameras = 0
        removed_detections = 0

        for cam_id in MOCK_CAMERA_IDS:
            # Check camera exists
            result = await session.execute(
                select(Camera).where(Camera.id == cam_id)
            )
            camera = result.scalar_one_or_none()

            if not camera:
                print(f"   ⏭️  {cam_id} — not found, skipping")
                continue

            # Count and delete its detections first
            det_result = await session.execute(
                select(Detection).where(Detection.camera_id == cam_id)
            )
            detections = det_result.scalars().all()
            det_count = len(detections)

            for det in detections:
                await session.delete(det)

            # Delete the camera
            await session.delete(camera)
            await session.commit()

            print(f"   ✅ Removed {cam_id} ({camera.name}) — {det_count} detections deleted")
            removed_cameras += 1
            removed_detections += det_count

        print(f"\n{'=' * 55}")
        print(f"   Cameras removed  : {removed_cameras}")
        print(f"   Detections removed: {removed_detections}")

        # Show what's left
        remaining = await session.execute(select(Camera).where(Camera.is_active == True))
        real_cameras = remaining.scalars().all()

        print(f"\n   Real cameras remaining: {len(real_cameras)}")
        for cam in real_cameras[:5]:
            print(f"   📷 {cam.name[:50]}")
        if len(real_cameras) > 5:
            print(f"   ... and {len(real_cameras) - 5} more")

        print(f"\n   Restart backend and refresh dashboard to see only real data.\n")


if __name__ == "__main__":
    asyncio.run(remove_mock_data())