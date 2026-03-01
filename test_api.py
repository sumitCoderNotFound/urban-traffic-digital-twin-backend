import requests

print("Testing Urban Observatory API endpoints...\n")

endpoints = [
    "https://urbanobservatory.ac.uk/api/cameras/latest",
    "https://urbanobservatory.ac.uk/api/cams/latest",
    "https://api.newcastle.urbanobservatory.ac.uk/api/v2/sensors/json/",
]

for url in endpoints:
    try:
        print(f"Testing: {url}")
        r = requests.get(url, timeout=10)
        print(f"   Status: {r.status_code}")
        if r.status_code == 200 and 'json' in r.headers.get('content-type', ''):
            data = r.json()
            if isinstance(data, list):
                print(f"   SUCCESS! Found {len(data)} items")
            elif isinstance(data, dict):
                print(f"   SUCCESS! Keys: {list(data.keys())}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
