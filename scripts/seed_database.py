"""
Database Seed Script
Initializes the database with Newcastle traffic cameras

Run: python scripts/seed_database.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import async_session_maker, engine, Base
from app.models.models import Camera, Detection
from datetime import datetime, timedelta
import random


# Newcastle traffic camera locations (matching frontend)
NEWCASTLE_CAMERAS = [
    {
        "id": "cam-001",
        "name": "Grey Street",
        "latitude": 54.9714,
        "longitude": -1.6120,
        "area": "City Centre",
        "description": "Main shopping street in Newcastle city centre"
    },
    {
        "id": "cam-002",
        "name": "Quayside",
        "latitude": 54.9695,
        "longitude": -1.6037,
        "area": "Quayside",
        "description": "Newcastle Quayside area near the Tyne Bridge"
    },
    {
        "id": "cam-003",
        "name": "Central Station",
        "latitude": 54.9686,
        "longitude": -1.6174,
        "area": "City Centre",
        "description": "Newcastle Central Railway Station entrance"
    },
    {
        "id": "cam-004",
        "name": "Jesmond Road",
        "latitude": 54.9812,
        "longitude": -1.5987,
        "area": "Jesmond",
        "description": "Main road through Jesmond residential area"
    },
    {
        "id": "cam-005",
        "name": "Gateshead Millennium Bridge",
        "latitude": 54.9697,
        "longitude": -1.5993,
        "area": "Gateshead",
        "description": "Iconic tilting bridge connecting Newcastle and Gateshead"
    },
    {
        "id": "cam-006",
        "name": "Haymarket",
        "latitude": 54.9784,
        "longitude": -1.6148,
        "area": "City Centre",
        "description": "Haymarket bus station and metro area"
    },
    {
        "id": "cam-007",
        "name": "Eldon Square",
        "latitude": 54.9753,
        "longitude": -1.6145,
        "area": "City Centre",
        "description": "Eldon Square shopping centre entrance"
    },
    {
        "id": "cam-008",
        "name": "St James Park",
        "latitude": 54.9756,
        "longitude": -1.6217,
        "area": "City Centre",
        "description": "Near Newcastle United football stadium"
    },
    {
        "id": "cam-009",
        "name": "Byker Bridge",
        "latitude": 54.9720,
        "longitude": -1.5850,
        "area": "Byker",
        "description": "Major route connecting Byker to city centre"
    },
    {
        "id": "cam-010",
        "name": "Tyne Bridge",
        "latitude": 54.9680,
        "longitude": -1.6070,
        "area": "Quayside",
        "description": "Iconic Tyne Bridge connecting Newcastle and Gateshead"
    },
]


def calculate_traffic_level(vehicles: int, pedestrians: int) -> str:
    """Calculate traffic level based on counts"""
    if vehicles > 40 or pedestrians > 150:
        return "high"
    elif vehicles > 20 or pedestrians > 80:
        return "medium"
    return "low"


async def seed_database():
    """Seed the database with sample data"""
    
    print("🌱 Starting database seed...")
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created")
    
    async with async_session_maker() as session:
        # Clear existing data
        await session.execute(Detection.__table__.delete())
        await session.execute(Camera.__table__.delete())
        await session.commit()
        print("🗑️  Cleared existing data")
        
        # Add cameras
        for cam_data in NEWCASTLE_CAMERAS:
            camera = Camera(
                id=cam_data["id"],
                name=cam_data["name"],
                latitude=cam_data["latitude"],
                longitude=cam_data["longitude"],
                area=cam_data["area"],
                description=cam_data["description"],
                status="online" if cam_data["id"] != "cam-006" else "offline",  # One offline camera
            )
            session.add(camera)
        
        await session.commit()
        print(f"✅ Added {len(NEWCASTLE_CAMERAS)} cameras")
        
        # Generate sample detections for the last 24 hours
        now = datetime.utcnow()
        detection_count = 0
        
        for camera in NEWCASTLE_CAMERAS:
            if camera["id"] == "cam-006":  # Skip offline camera
                continue
            
            # Generate detections every 15 minutes for last 24 hours
            for hours_ago in range(24):
                for minutes in [0, 15, 30, 45]:
                    timestamp = now - timedelta(hours=hours_ago, minutes=minutes)
                    
                    # Simulate traffic patterns (higher during rush hours)
                    hour = timestamp.hour
                    if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
                        multiplier = 1.5
                    elif 22 <= hour or hour <= 5:  # Night
                        multiplier = 0.3
                    else:
                        multiplier = 1.0
                    
                    # Random counts with area-based variation
                    base_vehicles = random.randint(10, 50)
                    base_pedestrians = random.randint(20, 150)
                    base_cyclists = random.randint(2, 20)
                    
                    vehicles = int(base_vehicles * multiplier)
                    pedestrians = int(base_pedestrians * multiplier)
                    cyclists = int(base_cyclists * multiplier)
                    
                    detection = Detection(
                        camera_id=camera["id"],
                        vehicles=vehicles,
                        pedestrians=pedestrians,
                        cyclists=cyclists,
                        cars=int(vehicles * 0.7),
                        buses=int(vehicles * 0.1),
                        trucks=int(vehicles * 0.15),
                        motorcycles=int(vehicles * 0.05),
                        traffic_level=calculate_traffic_level(vehicles, pedestrians),
                        congestion_score=min((vehicles + pedestrians / 2) / 2, 100),
                        detected_at=timestamp,
                    )
                    session.add(detection)
                    detection_count += 1
        
        await session.commit()
        print(f"✅ Added {detection_count} sample detections")
    
    print("🎉 Database seeding complete!")


if __name__ == "__main__":
    asyncio.run(seed_database())
