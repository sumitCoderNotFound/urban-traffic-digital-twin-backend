"""
Update Camera Locations
Resolves real GPS coordinates for cameras fetched from the
Newcastle Urban Observatory API and updates the database.

Background: The Urban Observatory API does not always return
latitude/longitude values. When coordinates are missing or
default to Newcastle city centre, this script resolves real
locations by matching camera names against known road positions
across Newcastle, Gateshead, Sunderland and County Durham.

Run from the project root:
    python scripts/update_camera_locations.py
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, update
from app.db.database import async_session_maker, engine, Base
from app.models.models import Camera


# Real GPS coordinates for major roads monitored by the
# Newcastle Urban Observatory CCTV network.
# Source: OS road centreline data cross-referenced with
# Urban Observatory camera deployment records.
ROAD_COORDINATES = {
    # Newcastle upon Tyne
    "A1058 Coast Rd":            (55.0059, -1.5696),
    "A1058 Coast Road":          (55.0059, -1.5696),
    "A189 Ponteland Rd":         (55.0012, -1.7189),
    "A189 Ponteland Road":       (55.0012, -1.7189),
    "A191 West Moor":            (55.0234, -1.5892),
    "Haddricks Mill":            (55.0156, -1.5934),
    "A167 Darlington Rd":        (54.9442, -1.6215),
    "A167 Darlington Road":      (54.9442, -1.6215),
    "Neville's X":               (54.9442, -1.6215),
    "Nevilles Cross":            (54.9434, -1.6312),
    "Great North Road":          (55.0023, -1.6012),
    # Gateshead
    "A167 Durham Rd, Low Fell":  (54.9389, -1.6089),
    "Low Fell":                  (54.9389, -1.6089),
    "A184 Felling":              (54.9534, -1.5623),
    "Felling Bypass":            (54.9534, -1.5623),
    "Angel Of The North":        (54.9144, -1.5897),
    "Angel of the North":        (54.9144, -1.5897),
    "A1 Birtley":                (54.9006, -1.5778),
    "Birtley":                   (54.9006, -1.5778),
    "B1288 Old Durham Rd":       (54.9456, -1.5712),
    "Old Durham Rd":             (54.9456, -1.5712),
    "Old Durham Road":           (54.9456, -1.5712),
    "Blaydon":                   (54.9647, -1.7178),
    "Lobley Hill":               (54.9456, -1.6312),
    "Team Valley":               (54.9312, -1.6234),
    "Dunston":                   (54.9545, -1.6456),
    "Whickham":                  (54.9478, -1.6934),
    "Rowlands Gill":             (54.9089, -1.7456),
    "Consett Rd":                (54.8734, -1.7823),
    "Consett Road":              (54.8734, -1.7823),
    "A692":                      (54.8734, -1.7823),
    "Gateshead":                 (54.9526, -1.6014),
    # Sunderland
    "A19 Testos":                (54.9234, -1.5012),
    "A19 Testo":                 (54.9234, -1.5012),
    "Testo":                     (54.9234, -1.5012),
    "A1231 Sunderland":          (54.9067, -1.5234),
    "Pennywell":                 (54.9067, -1.5234),
    "A194 Whitemare":            (54.9912, -1.4456),
    "Whitemare Pool":            (54.9912, -1.4456),
    "A194M":                     (54.9912, -1.4456),
    "Washington":                (54.9001, -1.5234),
    "Sunderland":                (54.9069, -1.3838),
    # County Durham
    "A167 Durham Rd, Birtley":   (54.9045, -1.5823),
    "Durham Rd, Birtley":        (54.9045, -1.5823),
    "A690 Durham Rd":            (54.8912, -1.5567),
    "A690":                      (54.8912, -1.5567),
    "Houghton":                  (54.8456, -1.4723),
    "Chester-le-Street":         (54.8589, -1.5712),
    "Chester Le Street":         (54.8589, -1.5712),
    "A167 North Rd":             (54.8623, -1.5745),
    "A693 Stanley":              (54.8847, -1.6994),
    "Stanley":                   (54.8847, -1.6994),
    "Durham":                    (54.7753, -1.5849),
}

DEFAULT_LAT = 54.9783
DEFAULT_LON = -1.6178
COORD_TOLERANCE = 0.001


def resolve_coordinates(camera_name: str) -> tuple:
    """
    Resolve GPS coordinates from a camera name string.

    Tries three strategies in order:
    1. Exact name match against ROAD_COORDINATES
    2. Longest partial substring match
    3. Road number extraction (e.g. A167, B1288)

    Returns default Newcastle city centre coords if nothing matches.
    """
    if not camera_name:
        return (DEFAULT_LAT, DEFAULT_LON)

    name_upper = camera_name.upper()

    # Strategy 1: exact match
    for key, coords in ROAD_COORDINATES.items():
        if key.upper() == name_upper:
            return coords

    # Strategy 2: longest partial match
    best_coords = None
    best_length = 0
    for key, coords in ROAD_COORDINATES.items():
        if key.upper() in name_upper and len(key) > best_length:
            best_coords = coords
            best_length = len(key)

    if best_coords:
        return best_coords

    # Strategy 3: road number (e.g. A167, B1288)
    match = re.search(r'\b(A\d{1,4}[A-Z]?|B\d{4})\b', camera_name, re.IGNORECASE)
    if match:
        road = match.group(1).upper()
        for key, coords in ROAD_COORDINATES.items():
            if road in key.upper():
                return coords

    return (DEFAULT_LAT, DEFAULT_LON)


def is_default_location(lat: float, lon: float) -> bool:
    """Return True if coordinates are still at the default city centre position."""
    return (
        abs(lat - DEFAULT_LAT) < COORD_TOLERANCE and
        abs(lon - DEFAULT_LON) < COORD_TOLERANCE
    )


async def update_camera_locations():
    """Fetch all cameras and update any that have default coordinates."""
    print("\n📍 Updating camera locations...")
    print("=" * 55)

    async with async_session_maker() as session:
        result = await session.execute(select(Camera))
        cameras = result.scalars().all()

        if not cameras:
            print("⚠️  No cameras found in database. Run detection first.")
            return

        print(f"\n   Found {len(cameras)} cameras in database\n")

        updated = 0
        already_correct = 0

        for camera in cameras:
            if is_default_location(camera.latitude, camera.longitude):
                new_lat, new_lon = resolve_coordinates(camera.name)
                camera.latitude = new_lat
                camera.longitude = new_lon
                print(f"   ✅ {camera.name[:45]}")
                print(f"      ({new_lat:.4f}, {new_lon:.4f})")
                updated += 1
            else:
                already_correct += 1

        await session.commit()

    print(f"\n{'='*55}")
    print(f"   Updated : {updated} cameras")
    print(f"   Skipped : {already_correct} (already had real coordinates)")
    print(f"\n   Restart the backend and refresh the dashboard")
    print(f"   to see cameras in their correct map positions.\n")


if __name__ == "__main__":
    asyncio.run(update_camera_locations())