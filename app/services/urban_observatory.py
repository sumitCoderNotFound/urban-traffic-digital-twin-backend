import os
import re
import asyncio
import aiohttp
import requests
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class CameraInfo:
    """Information about a traffic camera"""
    camera_id: str
    name: str
    short_description: str
    place: str
    latitude: float
    longitude: float
    image_url: str
    timestamp: str


# ============================================================
# REAL CAMERA COORDINATES — Newcastle Urban Observatory
# Manually verified GPS coordinates for known camera locations
# across Newcastle, Gateshead, Sunderland and Durham regions.
# These are used as fallback when the API does not return lat/lon.
# ============================================================
CAMERA_COORDINATE_LOOKUP = {
    # ── Newcastle upon Tyne ──────────────────────────────────
    "A1058 Coast Rd":           (55.0059, -1.5696),
    "A1058 Coast Road":         (55.0059, -1.5696),
    "A189 Ponteland Rd":        (55.0012, -1.7189),
    "A189 Ponteland Road":      (55.0012, -1.7189),
    "A191 West Moor":           (55.0234, -1.5892),
    "Haddricks Mill":           (55.0156, -1.5934),
    "A167 Darlington Rd":       (54.9442, -1.6215),
    "A167 Darlington Road":     (54.9442, -1.6215),
    "Neville's X":              (54.9442, -1.6215),
    "Nevilles Cross":           (54.9434, -1.6312),
    "Great North Road":         (55.0023, -1.6012),
    "A1":                       (54.9006, -1.5778),

    # ── Gateshead ────────────────────────────────────────────
    "A167 Durham Rd, Low Fell": (54.9389, -1.6089),
    "Low Fell":                 (54.9389, -1.6089),
    "A184 Felling":             (54.9534, -1.5623),
    "A184 Felling Bypass":      (54.9534, -1.5623),
    "Felling Bypass":           (54.9534, -1.5623),
    "Angel Of The North":       (54.9144, -1.5897),
    "Angel of the North":       (54.9144, -1.5897),
    "A1 Birtley":               (54.9006, -1.5778),
    "Birtley":                  (54.9006, -1.5778),
    "B1288 Old Durham Rd":      (54.9456, -1.5712),
    "Old Durham Rd":            (54.9456, -1.5712),
    "Old Durham Road":          (54.9456, -1.5712),
    "Blaydon":                  (54.9647, -1.7178),
    "Lobley Hill":              (54.9456, -1.6312),
    "Team Valley":              (54.9312, -1.6234),
    "Dunston":                  (54.9545, -1.6456),
    "Whickham":                 (54.9478, -1.6934),
    "Rowlands Gill":            (54.9089, -1.7456),
    "A692":                     (54.8734, -1.7823),
    "Consett Rd":               (54.8734, -1.7823),
    "Consett Road":             (54.8734, -1.7823),
    "Gateshead":                (54.9526, -1.6014),

    # ── Sunderland ───────────────────────────────────────────
    "A19 Testos":               (54.9234, -1.5012),
    "A19 Testo":                (54.9234, -1.5012),
    "Testo":                    (54.9234, -1.5012),
    "A1231 Sunderland":         (54.9067, -1.5234),
    "Pennywell":                (54.9067, -1.5234),
    "A194 Whitemare":           (54.9912, -1.4456),
    "Whitemare Pool":           (54.9912, -1.4456),
    "A194M":                    (54.9912, -1.4456),
    "Washington":               (54.9001, -1.5234),
    "Sunderland":               (54.9069, -1.3838),

    # ── County Durham ────────────────────────────────────────
    "A167 Durham Rd, Birtley":  (54.9045, -1.5823),
    "Durham Rd, Birtley":       (54.9045, -1.5823),
    "A690 Durham Rd":           (54.8912, -1.5567),
    "A690":                     (54.8912, -1.5567),
    "Houghton":                 (54.8456, -1.4723),
    "Chester-le-Street":        (54.8589, -1.5712),
    "Chester Le Street":        (54.8589, -1.5712),
    "A167 North Rd":            (54.8623, -1.5745),
    "Stanley":                  (54.8847, -1.6994),
    "A693 Stanley":             (54.8847, -1.6994),
    "Durham":                   (54.7753, -1.5849),

    # ── Default fallback ─────────────────────────────────────
    "Newcastle":                (54.9783, -1.6178),
}


def lookup_coordinates(camera_name: str) -> tuple:
    """
    Look up real GPS coordinates for a camera by its name.

    Strategy:
    1. Try exact match
    2. Try partial keyword match (longest matching key wins)
    3. Fall back to Newcastle city centre

    Args:
        camera_name: Camera name string from Urban Observatory API

    Returns:
        (latitude, longitude) tuple
    """
    if not camera_name:
        return (54.9783, -1.6178)

    name_upper = camera_name.upper()

    # 1. Exact match (case-insensitive)
    for key, coords in CAMERA_COORDINATE_LOOKUP.items():
        if key.upper() == name_upper:
            return coords

    # 2. Partial match — find the longest key that appears in the camera name
    best_match = None
    best_len = 0
    for key, coords in CAMERA_COORDINATE_LOOKUP.items():
        if key.upper() in name_upper and len(key) > best_len:
            best_match = coords
            best_len = len(key)

    if best_match:
        return best_match

    # 3. Road number match (e.g. extract "A167" and match)
    road_match = re.search(r'\b(A\d{1,4}[A-Z]?|B\d{4})\b', camera_name, re.IGNORECASE)
    if road_match:
        road_num = road_match.group(1).upper()
        for key, coords in CAMERA_COORDINATE_LOOKUP.items():
            if road_num in key.upper():
                return coords

    # 4. Fallback: Newcastle city centre
    return (54.9783, -1.6178)


class UrbanObservatoryCollector:
    """
    Collects REAL traffic camera images from Newcastle Urban Observatory.

    Dashboard: https://urbanobservatory.ac.uk/dash/cams/
    API: https://portal.cctv.urbanobservatory.ac.uk/latest

    The dashboard fetches camera data with image paths that need to be decoded:
    - safe_photo_path: "=slash-data-slash-photos_2-slash-NC_A167A1-slash-..."
    - Decodes to: "/data/photos_2/NC_A167A1/..."
    - Full URL: "https://portal.cctv.urbanobservatory.ac.uk/photo/{safe_photo_path}"
    """

    # Correct working API URL (verified 01/03/2026)
    CAMERA_LIST_URL = "https://portal.cctv.urbanobservatory.ac.uk/latest"
    PHOTO_BASE_URL = "https://portal.cctv.urbanobservatory.ac.uk/photo"

    def __init__(self):
        """Initialise the collector"""
        self.timeout = aiohttp.ClientTimeout(total=settings.URBAN_OBSERVATORY_TIMEOUT)
        self.image_storage_path = Path(settings.IMAGE_STORAGE_PATH)
        self.image_storage_path.mkdir(parents=True, exist_ok=True)

        self.headers = {
            'User-Agent': 'UrbanDigitalTwin/1.0 (MSc Research Project - Northumbria University)',
            'Accept': 'application/json, text/html, */*',
        }

        self.cameras_cache = []

    async def fetch_cameras_from_api(self) -> List[Dict]:
        """Fetch camera list from Urban Observatory portal."""
        print("📡 Fetching cameras from Newcastle Urban Observatory...")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    self.CAMERA_LIST_URL,
                    headers=self.headers,
                    ssl=False  # UO uses self-signed cert
                ) as response:
                    if response.status == 200:
                        data = await response.json(content_type=None)

                        cameras = []
                        items = data if isinstance(data, list) else data.get('cameras', data.get('items', []))

                        for item in items:
                            camera = self._parse_camera_item(item)
                            if camera:
                                cameras.append(camera)

                        print(f"   ✅ Found {len(cameras)} cameras")
                        self.cameras_cache = cameras
                        return cameras
                    else:
                        print(f"   ❌ HTTP {response.status}")
                        return []
        except Exception as e:
            print(f"   ❌ API error: {e}")
            return []

    def _parse_camera_item(self, item: Dict) -> Optional[Dict]:
        """
        Parse a camera item from the Urban Observatory API response.

        Coordinate resolution order:
        1. Use lat/lon from API response if valid (not 0.0 or missing)
        2. Fall back to lookup_coordinates() using camera name
        """
        try:
            safe_path = item.get('safe_photo_path', '')

            # Build image URL — keep safe_photo_path as-is (do NOT decode slashes)
            image_url = f"{self.PHOTO_BASE_URL}/{safe_path}" if safe_path else ''

            camera_name = item.get('short_description', item.get('name', ''))

            camera_id = (
                item.get('id') or
                item.get('camera_id') or
                camera_name.replace(' ', '_').replace('/', '_') or
                f"camera_{len(self.cameras_cache)}"
            )

            # ── Coordinate resolution ──────────────────────────
            api_lat = item.get('lat', item.get('latitude', None))
            api_lon = item.get('lon', item.get('longitude', None))

            # Use API coords only if they are valid (non-zero, non-null)
            if api_lat and api_lon and float(api_lat) != 0.0 and float(api_lon) != 0.0:
                latitude = float(api_lat)
                longitude = float(api_lon)
            else:
                # Fall back to name-based lookup
                latitude, longitude = lookup_coordinates(camera_name)

            return {
                'camera_id': camera_id,
                'name': camera_name or camera_id,
                'short_description': camera_name,
                'place': item.get('place', 'Newcastle'),
                'latitude': latitude,
                'longitude': longitude,
                'image_url': image_url,
                'safe_photo_path': safe_path,
                'timestamp': item.get('timestamp', datetime.utcnow().isoformat()),
            }
        except Exception as e:
            print(f"   ⚠️ Failed to parse camera: {e}")
            return None

    async def get_camera_sensors(self) -> List[Dict]:
        """Get list of cameras (main entry point)."""
        cameras = await self.fetch_cameras_from_api()
        return cameras

    async def fetch_camera_image(self, image_url: str, camera_id: str) -> Optional[str]:
        """
        Fetch and save a camera image.

        Args:
            image_url: URL of the camera image
            camera_id: Camera identifier for filename

        Returns:
            Path to saved image or None if failed
        """
        if not image_url:
            print(f"   ⚠️ No image URL for {camera_id}")
            return None

        safe_id = re.sub(r'[^\w\-]', '_', camera_id)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_id}_{timestamp}.jpg"
        filepath = self.image_storage_path / filename

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(image_url, headers=self.headers, ssl=False) as response:
                    if response.status == 200:
                        content = await response.read()

                        if len(content) < 5000:
                            print(f"   ⚠️ Image too small ({len(content)} bytes): {camera_id}")
                            return None

                        content_type = response.headers.get('content-type', '')
                        if 'image' not in content_type and 'octet' not in content_type:
                            print(f"   ⚠️ Not an image ({content_type}): {camera_id}")
                            return None

                        with open(filepath, "wb") as f:
                            f.write(content)

                        print(f"   ✅ Saved: {filepath.name} ({len(content)//1024}KB)")
                        return str(filepath)
                    else:
                        print(f"   ❌ HTTP {response.status}: {camera_id}")
                        return None

        except asyncio.TimeoutError:
            print(f"   ⏱️ Timeout: {camera_id}")
            return None
        except Exception as e:
            print(f"   ❌ Error fetching {camera_id}: {e}")
            return None

    async def collect_all_images(self) -> List[Dict]:
        """
        Collect images from all available cameras.

        Returns:
            List of dicts with camera_id, image_path, coordinates, and metadata
        """
        cameras = await self.get_camera_sensors()

        if not cameras:
            print("❌ No cameras found")
            return []

        print(f"\n📥 Downloading images from {len(cameras)} cameras...\n")

        collected = []

        for camera in cameras:
            image_path = await self.fetch_camera_image(
                camera.get('image_url', ''),
                camera.get('camera_id', 'unknown')
            )

            if image_path:
                collected.append({
                    'camera_id': camera['camera_id'],
                    'camera_name': camera.get('name', camera['camera_id']),
                    'short_description': camera.get('short_description', ''),
                    'place': camera.get('place', 'Newcastle'),
                    'image_path': image_path,
                    'image_url': camera.get('image_url', ''),
                    'latitude': camera.get('latitude', 54.9783),
                    'longitude': camera.get('longitude', -1.6178),
                    'timestamp': datetime.utcnow().isoformat(),
                })

        print(f"\n✅ Collected {len(collected)}/{len(cameras)} images")
        return collected


class DataCollectionScheduler:
    """Schedules periodic data collection from Urban Observatory."""

    def __init__(self, interval_seconds: int = None):
        self.interval = interval_seconds or settings.COLLECTION_INTERVAL
        self.collector = UrbanObservatoryCollector()
        self.running = False
        self.detector = None

    def _get_detector(self):
        if self.detector is None:
            from app.services.yolo_detector import get_detector
            self.detector = get_detector()
        return self.detector

    async def run_collection_cycle(self) -> List[Dict]:
        """Run a single collection and detection cycle"""
        print("\n" + "="*60)
        print(f"📸 COLLECTION CYCLE: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        collected = await self.collector.collect_all_images()

        if not collected:
            print("⚠️ No images collected")
            return []

        detector = self._get_detector()
        results = []

        print(f"\n🤖 Running YOLO detection on {len(collected)} images...\n")

        for item in collected:
            image_path = item['image_path']
            detection = detector.detect(image_path)

            if detection:
                result = {
                    **item,
                    'vehicles': detection.vehicles,
                    'pedestrians': detection.pedestrians,
                    'cyclists': detection.cyclists,
                    'cars': detection.cars,
                    'buses': detection.buses,
                    'trucks': detection.trucks,
                    'motorcycles': detection.motorcycles,
                    'confidence_avg': detection.confidence_avg,
                    'processing_time_ms': detection.processing_time_ms,
                }
                results.append(result)

                print(f"   📊 {item['camera_name'][:30]}: "
                      f"V:{detection.vehicles} P:{detection.pedestrians} C:{detection.cyclists}")

        print(f"\n✅ Detection complete: {len(results)} results")
        return results

    async def start(self):
        """Start the collection scheduler"""
        self.running = True
        print(f"\n🚀 Starting data collection scheduler")
        print(f"   Interval: {self.interval} seconds")

        while self.running:
            try:
                await self.run_collection_cycle()
            except Exception as e:
                print(f"❌ Collection error: {e}")

            if self.running:
                print(f"\n⏳ Next collection in {self.interval} seconds...")
                await asyncio.sleep(self.interval)

    def stop(self):
        self.running = False
        print("\n🛑 Data collection stopped")


# Utility functions
async def test_api_connection() -> bool:
    collector = UrbanObservatoryCollector()
    cameras = await collector.get_camera_sensors()
    if cameras:
        print(f"\n✅ API connected. Found {len(cameras)} cameras.")
        # Show sample coordinates
        for cam in cameras[:5]:
            print(f"   📍 {cam['name'][:40]} → ({cam['latitude']:.4f}, {cam['longitude']:.4f})")
        return True
    else:
        print("\n❌ Failed to connect to Urban Observatory API")
        return False


async def list_available_cameras():
    collector = UrbanObservatoryCollector()
    cameras = await collector.get_camera_sensors()

    if not cameras:
        print("\n❌ No cameras found")
        return []

    print(f"\n📷 Found {len(cameras)} cameras:\n")
    for i, cam in enumerate(cameras, 1):
        print(f"  {i:3}. {cam.get('name', 'Unknown')[:45]}")
        print(f"        GPS: ({cam.get('latitude'):.4f}, {cam.get('longitude'):.4f})")

    return cameras


async def collect_and_detect():
    scheduler = DataCollectionScheduler()
    results = await scheduler.run_collection_cycle()

    if results:
        total_vehicles = sum(r.get('vehicles', 0) for r in results)
        total_pedestrians = sum(r.get('pedestrians', 0) for r in results)
        total_cyclists = sum(r.get('cyclists', 0) for r in results)

        print(f"\n   Cameras processed: {len(results)}")
        print(f"   Total vehicles: {total_vehicles}")
        print(f"   Total pedestrians: {total_pedestrians}")
        print(f"   Total cyclists: {total_cyclists}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "test":
            asyncio.run(test_api_connection())
        elif cmd == "list":
            asyncio.run(list_available_cameras())
        elif cmd == "collect":
            asyncio.run(collect_and_detect())
        else:
            print(f"Unknown command: {cmd}")
    else:
        asyncio.run(test_api_connection())