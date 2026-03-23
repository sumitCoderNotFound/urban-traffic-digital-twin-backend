"""
Urban Observatory Camera Data Collector
========================================
Fetches REAL camera images from Newcastle Urban Observatory

Author: Sumit Malviya
Supervisor: Dr. Jason Moore
Northumbria University
"""

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
# COORDINATE CACHE — pre-resolved for known UO camera locations.
# Used as instant fallback when Nominatim geocoding is unavailable
# or for cameras whose place names are ambiguous road numbers.
# ============================================================
COORDINATE_CACHE = {
    "A1058 Coast Rd":                (55.0059, -1.5696),
    "A1058 Coast Road":              (55.0059, -1.5696),
    "A189 Ponteland Rd":             (55.0012, -1.7189),
    "A189 Ponteland Road":           (55.0012, -1.7189),
    "A191 West Moor":                (55.0234, -1.5892),
    "Haddricks Mill":                (55.0156, -1.5934),
    "Great North Road":              (55.0023, -1.6012),
    "A167 Darlington Rd":            (54.9442, -1.6215),
    "A167 Darlington Road":          (54.9442, -1.6215),
    "Neville's X":                  (54.9442, -1.6215),
    "A167 Durham Rd, Low Fell":      (54.9389, -1.6089),
    "Low Fell":                      (54.9389, -1.6089),
    "A184 Felling":                  (54.9534, -1.5623),
    "Felling Bypass":                (54.9534, -1.5623),
    "Angel Of The North":            (54.9144, -1.5897),
    "Angel of the North":            (54.9144, -1.5897),
    "A1 Birtley":                    (54.9006, -1.5778),
    "Birtley":                       (54.9006, -1.5778),
    "B1288 Old Durham Rd":           (54.9456, -1.5712),
    "Blaydon":                       (54.9647, -1.7178),
    "Lobley Hill":                   (54.9456, -1.6312),
    "Team Valley":                   (54.9312, -1.6234),
    "Dunston":                       (54.9545, -1.6456),
    "Whickham":                      (54.9478, -1.6934),
    "Rowlands Gill":                 (54.9089, -1.7456),
    "Gateshead":                     (54.9526, -1.6014),
    "A19 Testos":                    (54.9234, -1.5012),
    "Testo":                         (54.9234, -1.5012),
    "Pennywell":                     (54.9067, -1.5234),
    "Whitemare Pool":                (54.9912, -1.4456),
    "Washington":                    (54.9001, -1.5234),
    "Sunderland":                    (54.9069, -1.3838),
    "A167 Cock O' the North":       (54.8334, -1.5723),
    "Cock O' the North":            (54.8334, -1.5723),
    "Duke of Wellington":            (54.8923, -1.5701),
    "A167 Durham Rd, Birtley":       (54.9045, -1.5823),
    "A690 Durham Rd":                (54.8912, -1.5567),
    "Houghton":                      (54.8456, -1.4723),
    "Chester-le-Street":             (54.8589, -1.5712),
    "Chester le Street":             (54.8589, -1.5712),
    "Stanley":                       (54.8847, -1.6994),
    "Shincliffe":                    (54.7612, -1.5423),
    "Durham":                        (54.7753, -1.5849),
    "Darlington":                    (54.5236, -1.5599),
    "Newcastle":                     (54.9783, -1.6178),
}

DEFAULT_COORDS = (54.9783, -1.6178)


class GeocoderCache:
    """
    Resolves GPS coordinates for camera place names.

    Strategy (in order):
    1. In-memory runtime cache — avoids repeat API calls within a session
    2. Pre-built COORDINATE_CACHE — instant results for known UO locations
    3. Nominatim (OpenStreetMap) geocoding — for any new/unknown place names,
       constrained to the North East England bounding box
    4. Road number fallback — extracts e.g. "A167" and finds nearest match
    5. Default — Newcastle city centre (54.9783, -1.6178)

    Nominatim is free and requires no API key. Requests are rate-limited
    to 1/second per OSM usage policy.
    """

    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    # Bounding box: North East England (minLon, minLat, maxLon, maxLat)
    BBOX = "-2.2,54.4,-1.2,55.2"

    def __init__(self):
        self._cache: Dict[str, tuple] = {}

    def _cache_key(self, name: str) -> str:
        return name.strip().upper()

    def _lookup_static(self, name: str) -> Optional[tuple]:
        """Check COORDINATE_CACHE with exact and partial matching."""
        name_upper = name.upper()

        # Exact match
        for key, coords in COORDINATE_CACHE.items():
            if key.upper() == name_upper:
                return coords

        # Longest partial match
        best, best_len = None, 0
        for key, coords in COORDINATE_CACHE.items():
            if key.upper() in name_upper and len(key) > best_len:
                best, best_len = coords, len(key)

        return best

    def _lookup_road_number(self, name: str) -> Optional[tuple]:
        """Extract road number (e.g. A167) and find nearest static entry."""
        match = re.search(r'\b(A\d{1,4}[A-Z]?|B\d{4})\b', name, re.IGNORECASE)
        if not match:
            return None
        road = match.group(1).upper()
        for key, coords in COORDINATE_CACHE.items():
            if road in key.upper():
                return coords
        return None

    async def geocode(self, place_name: str) -> tuple:
        """
        Resolve GPS coordinates for a place name.
        Uses Nominatim geocoding with COORDINATE_CACHE as fast fallback.
        """
        if not place_name or not place_name.strip():
            return DEFAULT_COORDS

        key = self._cache_key(place_name)

        # 1. Runtime cache
        if key in self._cache:
            return self._cache[key]

        # 2. Static cache
        static = self._lookup_static(place_name)
        if static:
            self._cache[key] = static
            return static

        # 3. Nominatim geocoding
        coords = await self._nominatim_geocode(place_name)
        if coords:
            self._cache[key] = coords
            print(f"   🗺️  Geocoded: {place_name[:45]} → {coords}")
            return coords

        # 4. Road number fallback
        road_fallback = self._lookup_road_number(place_name)
        if road_fallback:
            self._cache[key] = road_fallback
            return road_fallback

        # 5. Default
        self._cache[key] = DEFAULT_COORDS
        return DEFAULT_COORDS

    async def _nominatim_geocode(self, place_name: str) -> Optional[tuple]:
        """Call Nominatim API to geocode a place name within NE England."""
        # Clean up the query — remove truncation artifacts
        query = re.sub(r'\bRbt\b', 'Roundabout', place_name)
        query = f"{query}, England"

        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "viewbox": self.BBOX,
            "bounded": 1,
            "countrycodes": "gb",
        }

        headers = {
            "User-Agent": "UrbanDigitalTwin/1.0 (MSc Research - Northumbria University)",
            "Accept-Language": "en",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.NOMINATIM_URL,
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        if results:
                            lat = float(results[0]["lat"])
                            lon = float(results[0]["lon"])
                            # Sanity check — must be in North East England
                            if 54.3 < lat < 55.3 and -2.3 < lon < -1.0:
                                return (round(lat, 6), round(lon, 6))
        except Exception:
            pass  # Fall through to static fallback

        return None


# Module-level geocoder instance (shared across all requests)
_geocoder = GeocoderCache()


async def resolve_coordinates(place_name: str) -> tuple:
    """Public interface — resolve GPS coords for a camera place name."""
    return await _geocoder.geocode(place_name)


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
                            camera = await self._parse_camera_item(item)
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

    async def _parse_camera_item(self, item: Dict) -> Optional[Dict]:
        """
        Parse a camera item from the Urban Observatory API response.

        Coordinate resolution order:
        1. Use lat/lon from API response if valid (not 0.0 or missing)
        2. Fall back to resolve_coordinates() using Nominatim geocoding
        """
        try:
            safe_path = item.get('safe_photo_path', '')

            # Build image URL — keep safe_photo_path as-is (do NOT decode slashes)
            image_url = f"{self.PHOTO_BASE_URL}/{safe_path}" if safe_path else ''

            # UO API: 'place' is full road name, 'short_description' is truncated
            place       = item.get('place', '')
            short_desc  = item.get('short_description', item.get('name', ''))
            camera_name = place or short_desc

            camera_id = (
                item.get('id') or
                item.get('camera_id') or
                camera_name.replace(' ', '_').replace('/', '_') or
                f"camera_{len(self.cameras_cache)}"
            )

            # UO API does not return lat/lon — resolve from place name
            api_lat = item.get('lat', item.get('latitude', None))
            api_lon = item.get('lon', item.get('longitude', None))

            if api_lat and api_lon and float(api_lat) != 0.0 and float(api_lon) != 0.0:
                latitude = float(api_lat)
                longitude = float(api_lon)
            else:
                # Try full place name first (more complete than short_description)
                latitude, longitude = await resolve_coordinates(place or short_desc)

            return {
                'camera_id': camera_id,
                'name': camera_name or camera_id,
                'short_description': short_desc,
                'place': place or 'Newcastle',
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