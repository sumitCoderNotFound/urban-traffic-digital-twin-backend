#!/bin/bash
cd /Users/cashify/Downloads/Assignments\(INTERSHIP\)/MSC\ PROJECT\(DISERTATION\)/project/urban-digital-twin-backend
source venv/bin/activate

echo "Waiting until 10:30 AM..."
while [ $(date +%H%M) -lt 1030 ]; do sleep 60; done

echo "Starting midday collection at $(date)"
mkdir -p data/uo_dataset_midday

for i in $(seq 1 24); do
    echo "Round $i/24 at $(date)"
    python3 -c "
import asyncio, aiohttp, os, re
from datetime import datetime
from pathlib import Path

CAMERA_URL = 'https://portal.cctv.urbanobservatory.ac.uk/latest'
BASE_URL = 'https://portal.cctv.urbanobservatory.ac.uk'
BASE_DIR = Path('data/uo_dataset_midday')
BASE_DIR.mkdir(parents=True, exist_ok=True)

async def collect():
    async with aiohttp.ClientSession() as session:
        async with session.get(CAMERA_URL) as resp:
            cameras = await resp.json()
        print(f'  Found {len(cameras)} cameras')
        saved = 0
        for cam in cameras:
            try:
                name = cam.get('place', 'unknown')
                clean = re.sub(r'[^\w\s-]', '', name)
                clean = re.sub(r'\s+', '_', clean.strip())[:40]
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f'{clean}__{ts}.jpg'
                photo_path = cam.get('safe_photo_path', '').replace('~slash~', '/')
                url = BASE_URL + photo_path
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as img_resp:
                    if img_resp.status == 200:
                        data = await img_resp.read()
                        if len(data) > 5000:
                            (BASE_DIR / filename).write_bytes(data)
                            saved += 1
            except: pass
        print(f'  Saved {saved} images')

asyncio.run(collect())
"
    echo "Sleeping 5 minutes..."
    sleep 300
done

echo "Collection done at $(date). Running auto_label..."
python scripts/auto_label_v2.py --images data/uo_dataset_midday --conf 0.25 --device cpu --results data/results_midday
echo "All done at $(date)"
