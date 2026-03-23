import sys


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "test":
            # Test Urban Observatory API connection
            print("Testing Urban Observatory API connection...\n")
            import requests
            import warnings
            warnings.filterwarnings('ignore')

            r = requests.get(
                "https://portal.cctv.urbanobservatory.ac.uk/latest",
                verify=False,
                timeout=30
            )
            if r.status_code == 200:
                cameras = r.json()
                print(f"✅ API working — {len(cameras)} cameras available")
                print(f"   First camera: {cameras[0]['place']}")
                print(f"   Timestamp: {cameras[0]['timestamp']}")
            else:
                print(f"❌ API failed: {r.status_code}")

        elif command == "collect":
            # Collect images and run YOLO detection once
            print("Collecting images and running YOLO detection...\n")
            from app.services.urban_observatory import get_collector
            collector = get_collector()
            results = collector.run_collection_cycle(max_cameras=10)

            if results:
                print(f"\n✅ Processed {len(results)} cameras")
                total_vehicles = sum(r['vehicles'] for r in results)
                total_pedestrians = sum(r['pedestrians'] for r in results)
                print(f"   Total vehicles detected: {total_vehicles}")
                print(f"   Total pedestrians detected: {total_pedestrians}")
            else:
                print("No results — check API connection")

        elif command == "detect":
            # Run YOLO on a specific image
            if len(sys.argv) < 3:
                print("Usage: python run.py detect <image_path>")
                sys.exit(1)

            image_path = sys.argv[2]
            print(f"Running YOLO detection on: {image_path}\n")

            from app.services.yolo_detector import get_detector
            detector = get_detector()
            result = detector.detect(image_path)

            if result:
                print("Detection Results:")
                print(f"  Vehicles:    {result.vehicles}")
                print(f"    Cars:      {result.cars}")
                print(f"    Buses:     {result.buses}")
                print(f"    Trucks:    {result.trucks}")
                print(f"    Motorcycles: {result.motorcycles}")
                print(f"  Pedestrians: {result.pedestrians}")
                print(f"  Cyclists:    {result.cyclists}")
                print(f"  Processing:  {result.processing_time_ms}ms")
                print(f"  Confidence:  {result.confidence_avg}")
            else:
                print("Detection failed")

        else:
            print(f"Unknown command: {command}")
            print("Usage: python run.py [test|collect|detect <image>]")
            sys.exit(1)

    else:
        # Start the API server
        import uvicorn
        from app.core.config import settings

        print("""
Urban Digital Twin - Backend API
Author: Sumit Malviya
Northumbria University

Docs:   http://localhost:8000/api/docs
Health: http://localhost:8000/api/health
Run pipeline: POST http://localhost:8000/api/detections/run
        """)

        uvicorn.run(
            "app.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
        )


if __name__ == "__main__":
    main()