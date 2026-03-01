#!/usr/bin/env python
"""
Urban Digital Twin - Run Script
Quick start script for development

Usage:
    python run.py              # Start API server
    python run.py seed         # Seed database with sample data
    python run.py test         # Test Urban Observatory API connection
    python run.py cameras      # List available cameras
    python run.py collect      # Collect images + run YOLO detection
    python run.py live         # Start continuous data collection
    python run.py detect <img> # Run YOLO on a specific image
"""

import sys
import asyncio


def show_help():
    """Show help message"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           Urban Digital Twin - Backend Commands                ║
╚═══════════════════════════════════════════════════════════════╝

USAGE:
    python run.py <command>

COMMANDS:
    (none)      Start the FastAPI server
    seed        Seed database with sample data
    test        Test Urban Observatory API connection
    cameras     List all available traffic cameras
    collect     Collect images and run YOLO detection (once)
    live        Start continuous data collection (every 5 min)
    detect      Run YOLO detection on a specific image
    help        Show this help message

EXAMPLES:
    python run.py                    # Start API server
    python run.py collect            # Fetch real cameras + detect
    python run.py detect image.jpg   # Detect on specific image
    python run.py live               # Continuous monitoring

API ENDPOINTS (when server running):
    http://localhost:8000/api/docs      Swagger UI
    http://localhost:8000/api/cameras   List cameras
    http://localhost:8000/api/health    Health check
""")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "seed":
            # Seed database with sample data
            print("🌱 Seeding database...")
            from scripts.seed_database import seed_database
            asyncio.run(seed_database())
            
        elif command == "test":
            # Test Urban Observatory API connection
            from app.services.urban_observatory import test_api_connection
            asyncio.run(test_api_connection())
            
        elif command == "cameras":
            # List available cameras from Urban Observatory
            from app.services.urban_observatory import list_available_cameras
            asyncio.run(list_available_cameras())
            
        elif command == "collect":
            # Collect images and run YOLO detection (one-time)
            print("📸 Collecting images and running detection...")
            from app.services.urban_observatory import collect_and_detect
            results = asyncio.run(collect_and_detect())
            
            if results:
                print(f"\n✅ Successfully processed {len(results)} cameras")
            else:
                print("\n⚠️ No results - check API connection")
            
        elif command == "live":
            # Start continuous data collection
            print("🔴 Starting LIVE data collection...")
            from app.services.urban_observatory import DataCollectionScheduler
            
            scheduler = DataCollectionScheduler()
            try:
                asyncio.run(scheduler.start())
            except KeyboardInterrupt:
                scheduler.stop()
                print("\n👋 Stopped")
            
        elif command == "detect":
            # Run YOLO detection on a specific image
            if len(sys.argv) < 3:
                print("❌ Please provide an image path")
                print("   Usage: python run.py detect <image_path>")
                sys.exit(1)
            
            image_path = sys.argv[2]
            print(f"🤖 Running YOLO detection on: {image_path}")
            
            from app.services.yolo_detector import get_detector
            detector = get_detector()
            result = detector.detect(image_path)
            
            if result:
                print("\n📊 Detection Results:")
                print(f"   🚗 Vehicles: {result.vehicles}")
                print(f"      - Cars: {result.cars}")
                print(f"      - Buses: {result.buses}")
                print(f"      - Trucks: {result.trucks}")
                print(f"      - Motorcycles: {result.motorcycles}")
                print(f"   🚶 Pedestrians: {result.pedestrians}")
                print(f"   🚴 Cyclists: {result.cyclists}")
                print(f"   ⏱️  Processing time: {result.processing_time_ms}ms")
                print(f"   📈 Avg confidence: {result.confidence_avg}")
            else:
                print("❌ Detection failed")
            
        elif command == "help" or command == "-h" or command == "--help":
            show_help()
            
        else:
            print(f"❌ Unknown command: {command}")
            show_help()
            sys.exit(1)
    else:
        # Start the API server
        import uvicorn
        from app.core.config import settings
        
        print("""
╔═══════════════════════════════════════════════════════════════╗
║           Urban Digital Twin - Backend API                     ║
║                                                                ║
║   Author: Sumit Malviya                                        ║
║   Supervisor: Dr. Jason Moore                                  ║
║   Northumbria University                                       ║
║                                                                ║
║   📚 Docs: http://localhost:8000/api/docs                      ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        uvicorn.run(
            "app.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
        )


if __name__ == "__main__":
    main()
