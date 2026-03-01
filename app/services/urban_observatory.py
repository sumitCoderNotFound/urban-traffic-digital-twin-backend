"""
Urban Observatory Camera Data Collector
========================================
Fetches REAL camera images from Newcastle Urban Observatory
Dashboard: https://urbanobservatory.ac.uk/dash/cams/

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


class UrbanObservatoryCollector:
    """
    Collects REAL traffic camera images from Newcastle Urban Observatory.
    
    Dashboard: https://urbanobservatory.ac.uk/dash/cams/
    API: https://urbanobservatory.ac.uk/api/...
    
    The dashboard fetches camera data with image paths that need to be decoded:
    - safe_photo_path: "=slash-data-slash-photos_2-slash-NC_A167A1-slash-..."
    - Decodes to: "/data/photos_2/NC_A167A1/..."
    - Full URL: "https://urbanobservatory.ac.uk/data/photos_2/NC_A167A1/..."
    """
    
    # Base URLs
    DASHBOARD_URL = "https://urbanobservatory.ac.uk/dash/cams/"
    BASE_URL = "https://urbanobservatory.ac.uk"
    OLD_API_URL = "https://api.newcastle.urbanobservatory.ac.uk/api/v2"
    
    # Camera API endpoints to try
    CAMERA_API_ENDPOINTS = [
        "https://urbanobservatory.ac.uk/api/cameras/latest",
        "https://urbanobservatory.ac.uk/api/cams/latest",
        "https://urbanobservatory.ac.uk/dash/cams/api/latest",
    ]
    
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
        self.working_api_url = None
    
    def decode_safe_photo_path(self, safe_path: str) -> str:
        """
        Decode the safe_photo_path from the API response.
        
        Example:
            Input: "=slash-data-slash-photos_2-slash-NC_A167A1-slash-20260220-slash-220800.jpg"
            Output: "/data/photos_2/NC_A167A1/20260220/220800.jpg"
        """
        if not safe_path:
            return ""
        
        # Remove leading = if present
        if safe_path.startswith("="):
            safe_path = safe_path[1:]
        
        # Replace encoded slashes
        decoded = safe_path.replace("slash-", "/").replace("-slash", "")
        
        # Clean up any double slashes
        decoded = re.sub(r'/+', '/', decoded)
        
        # Ensure it starts with /
        if not decoded.startswith("/"):
            decoded = "/" + decoded
        
        return decoded
    
    def build_image_url(self, safe_path: str) -> str:
        """Build full image URL from safe_photo_path"""
        decoded_path = self.decode_safe_photo_path(safe_path)
        return f"{self.BASE_URL}{decoded_path}"
    
    async def discover_api_endpoint(self) -> Optional[str]:
        """Discover the working camera API endpoint"""
        print("🔍 Discovering camera API endpoint...")
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Try known endpoints
            for endpoint in self.CAMERA_API_ENDPOINTS:
                try:
                    async with session.get(endpoint, headers=self.headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and (isinstance(data, list) or 'cameras' in data):
                                print(f"   ✅ Found working API: {endpoint}")
                                self.working_api_url = endpoint
                                return endpoint
                except Exception as e:
                    continue
            
            # Try to extract from dashboard page
            try:
                async with session.get(self.DASHBOARD_URL, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Look for API URLs in the JavaScript
                        api_patterns = [
                            r'fetch\(["\']([^"\']+/api/[^"\']+)["\']',
                            r'url\s*[:=]\s*["\']([^"\']+/api/[^"\']+)["\']',
                            r'endpoint\s*[:=]\s*["\']([^"\']+)["\']',
                        ]
                        
                        for pattern in api_patterns:
                            matches = re.findall(pattern, html)
                            for match in matches:
                                if 'cam' in match.lower() or 'photo' in match.lower():
                                    full_url = match if match.startswith('http') else f"{self.BASE_URL}{match}"
                                    print(f"   📋 Found potential API in HTML: {full_url}")
                                    
                                    # Test it
                                    try:
                                        async with session.get(full_url, headers=self.headers) as test_response:
                                            if test_response.status == 200:
                                                self.working_api_url = full_url
                                                return full_url
                                    except:
                                        continue
            except Exception as e:
                print(f"   ⚠️ Could not parse dashboard: {e}")
        
        print("   ❌ No working API endpoint found")
        return None
    
    async def fetch_cameras_from_api(self) -> List[Dict]:
        """Fetch camera data from the API"""
        print("📡 Fetching cameras from Urban Observatory...")
        
        if not self.working_api_url:
            await self.discover_api_endpoint()
        
        if not self.working_api_url:
            print("   ⚠️ Using fallback: scraping dashboard")
            return await self.scrape_dashboard()
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.working_api_url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        cameras = []
                        items = data if isinstance(data, list) else data.get('cameras', data.get('items', []))
                        
                        for item in items:
                            camera = self._parse_camera_item(item)
                            if camera:
                                cameras.append(camera)
                        
                        print(f"   ✅ Found {len(cameras)} cameras")
                        self.cameras_cache = cameras
                        return cameras
        except Exception as e:
            print(f"   ❌ API error: {e}")
        
        return []
    
    def _parse_camera_item(self, item: Dict) -> Optional[Dict]:
        """Parse a camera item from API response"""
        try:
            # Extract safe_photo_path and decode it
            safe_path = item.get('safe_photo_path', '')
            image_url = self.build_image_url(safe_path) if safe_path else ''
            
            # Extract camera ID from various possible fields
            camera_id = (
                item.get('id') or 
                item.get('camera_id') or 
                item.get('short_description', '').replace(' ', '_').replace('/', '_') or
                f"camera_{len(self.cameras_cache)}"
            )
            
            return {
                'camera_id': camera_id,
                'name': item.get('short_description', item.get('name', camera_id)),
                'short_description': item.get('short_description', ''),
                'place': item.get('place', 'Newcastle'),
                'latitude': float(item.get('lat', item.get('latitude', 54.9783))),
                'longitude': float(item.get('lon', item.get('longitude', -1.6178))),
                'image_url': image_url,
                'safe_photo_path': safe_path,
                'timestamp': item.get('timestamp', datetime.utcnow().isoformat()),
            }
        except Exception as e:
            print(f"   ⚠️ Failed to parse camera: {e}")
            return None
    
    async def scrape_dashboard(self) -> List[Dict]:
        """Fallback: Scrape camera info from dashboard HTML"""
        print("🔍 Scraping dashboard for cameras...")
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.DASHBOARD_URL, headers=self.headers) as response:
                    if response.status != 200:
                        return []
                    
                    html = await response.text()
                    
                    cameras = []
                    
                    # Find image URLs in the HTML
                    img_pattern = r'<img[^>]+src=["\']([^"\']+(?:\.jpg|\.png|\.jpeg)[^"\']*)["\']'
                    images = re.findall(img_pattern, html, re.IGNORECASE)
                    
                    for img_url in images:
                        if 'photo' in img_url.lower() or 'camera' in img_url.lower() or 'data/' in img_url:
                            full_url = img_url if img_url.startswith('http') else f"{self.BASE_URL}{img_url}"
                            
                            # Extract camera ID from URL
                            parts = img_url.split('/')
                            camera_id = parts[-2] if len(parts) > 2 else f"camera_{len(cameras)}"
                            
                            cameras.append({
                                'camera_id': camera_id,
                                'name': camera_id.replace('_', ' ').replace('-', ' ').title(),
                                'short_description': camera_id,
                                'place': 'Newcastle',
                                'latitude': 54.9783,
                                'longitude': -1.6178,
                                'image_url': full_url,
                                'timestamp': datetime.utcnow().isoformat(),
                            })
                    
                    print(f"   ✅ Found {len(cameras)} cameras from HTML")
                    self.cameras_cache = cameras
                    return cameras
                    
        except Exception as e:
            print(f"   ❌ Scraping error: {e}")
            return []
    
    async def get_camera_sensors(self) -> List[Dict]:
        """Get list of camera sensors (compatible with old API)"""
        cameras = await self.fetch_cameras_from_api()
        
        if not cameras:
            # Fallback to old API
            cameras = await self._fetch_from_old_api()
        
        return cameras
    
    async def _fetch_from_old_api(self) -> List[Dict]:
        """Fetch from old Urban Observatory API"""
        print("📡 Trying old Urban Observatory API...")
        
        url = f"{self.OLD_API_URL}/sensors/json/"
        params = {"theme": "Traffic"}
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        sensors = data if isinstance(data, list) else data.get('sensors', [])
                        
                        cameras = []
                        for sensor in sensors:
                            sensor_name = sensor.get('name', sensor.get('sensorName', ''))
                            if 'camera' in sensor_name.lower() or 'image' in str(sensor).lower():
                                cameras.append({
                                    'camera_id': sensor_name,
                                    'name': sensor_name.replace('_', ' ').title(),
                                    'short_description': sensor_name,
                                    'place': 'Newcastle',
                                    'latitude': sensor.get('lat', 54.9783),
                                    'longitude': sensor.get('lon', -1.6178),
                                    'image_url': sensor.get('data', {}).get('url', ''),
                                    'timestamp': datetime.utcnow().isoformat(),
                                })
                        
                        print(f"   ✅ Found {len(cameras)} cameras from old API")
                        return cameras
        except Exception as e:
            print(f"   ❌ Old API error: {e}")
        
        return []
    
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
        
        # Clean camera_id for filename
        safe_id = re.sub(r'[^\w\-]', '_', camera_id)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_id}_{timestamp}.jpg"
        filepath = self.image_storage_path / filename
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(image_url, headers=self.headers) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Verify it's a valid image (at least 5KB)
                        if len(content) < 5000:
                            print(f"   ⚠️ Image too small ({len(content)} bytes): {camera_id}")
                            return None
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        if 'image' not in content_type and 'octet' not in content_type:
                            print(f"   ⚠️ Not an image ({content_type}): {camera_id}")
                            return None
                        
                        # Save image
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
            List of dictionaries with camera_id, image_path, and metadata
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
    """
    Schedules periodic data collection from Urban Observatory.
    Collects images and runs YOLO detection.
    """
    
    def __init__(self, interval_seconds: int = None):
        """
        Initialise the scheduler.
        
        Args:
            interval_seconds: Collection interval in seconds (default from settings)
        """
        self.interval = interval_seconds or settings.COLLECTION_INTERVAL
        self.collector = UrbanObservatoryCollector()
        self.running = False
        self.detector = None
    
    def _get_detector(self):
        """Get YOLO detector instance"""
        if self.detector is None:
            from app.services.yolo_detector import get_detector
            self.detector = get_detector()
        return self.detector
    
    async def run_collection_cycle(self) -> List[Dict]:
        """Run a single collection and detection cycle"""
        print("\n" + "="*60)
        print(f"📸 COLLECTION CYCLE: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Collect images
        collected = await self.collector.collect_all_images()
        
        if not collected:
            print("⚠️ No images collected")
            return []
        
        # Run detection on each image
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
        print(f"   Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                results = await self.run_collection_cycle()
                
                # TODO: Store results in database
                # This will be handled by the API routes
                
            except Exception as e:
                print(f"❌ Collection error: {e}")
            
            if self.running:
                print(f"\n⏳ Next collection in {self.interval} seconds...")
                await asyncio.sleep(self.interval)
    
    def stop(self):
        """Stop the collection scheduler"""
        self.running = False
        print("\n🛑 Data collection stopped")


# Utility functions
async def test_api_connection() -> bool:
    """Test connection to Urban Observatory API"""
    print("\n" + "="*60)
    print("🔍 TESTING URBAN OBSERVATORY API CONNECTION")
    print("="*60)
    
    collector = UrbanObservatoryCollector()
    
    # Try to discover API
    api_url = await collector.discover_api_endpoint()
    
    # Fetch cameras
    cameras = await collector.get_camera_sensors()
    
    if cameras:
        print(f"\n✅ API connected. Found {len(cameras)} cameras.")
        return True
    else:
        print("\n❌ Failed to connect to Urban Observatory API")
        return False


async def list_available_cameras():
    """List all available cameras from Urban Observatory"""
    print("\n" + "="*60)
    print("📷 AVAILABLE TRAFFIC CAMERAS")
    print("="*60)
    
    collector = UrbanObservatoryCollector()
    cameras = await collector.get_camera_sensors()
    
    if not cameras:
        print("\n❌ No cameras found")
        return []
    
    print(f"\n📷 Found {len(cameras)} cameras:\n")
    
    for i, cam in enumerate(cameras, 1):
        print(f"  {i:2}. {cam.get('name', 'Unknown')}")
        print(f"      ID: {cam.get('camera_id', 'N/A')}")
        print(f"      Location: ({cam.get('latitude', 'N/A')}, {cam.get('longitude', 'N/A')})")
        print(f"      Place: {cam.get('place', 'N/A')}")
        if cam.get('image_url'):
            print(f"      Image: {cam['image_url'][:60]}...")
        print()
    
    return cameras


async def collect_and_detect():
    """Collect images and run detection (one-time)"""
    scheduler = DataCollectionScheduler()
    results = await scheduler.run_collection_cycle()
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
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
            print("Usage: python urban_observatory.py [test|list|collect]")
    else:
        # Default: test connection
        asyncio.run(test_api_connection())
