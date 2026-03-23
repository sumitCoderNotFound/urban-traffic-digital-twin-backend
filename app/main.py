from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import cameras, detections, metrics, health
from app.core.config import settings
from app.db.database import engine, Base
from app.api.routes import cameras, detections, metrics, health, annotate

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables ready")

    # Start automatic detection scheduler
    # Runs every 15 minutes — fetches images from Urban Observatory,
    # runs YOLOv8, saves results to DB and dataset folder
    from app.services.scheduler import scheduler
    await scheduler.start()

    print(f"🚀 Urban Digital Twin API running on {settings.API_HOST}:{settings.API_PORT}")
    print(f"📚 Docs: http://localhost:{settings.API_PORT}/api/docs")

    yield

    # Shutdown — stop the scheduler cleanly
    from app.services.scheduler import scheduler
    await scheduler.stop()
    print("👋 Urban Digital Twin API stopped")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
## Urban Digital Twin — Traffic Monitoring API

Real-time traffic state estimation using YOLOv8 and Newcastle Urban Observatory camera feeds.

### Features
- 📷 Live CCTV images from 345 Newcastle Urban Observatory cameras
- 🤖 YOLOv8 object detection — vehicles, pedestrians, cyclists
- ⏰ Automatic detection every 15 minutes via background scheduler
- 📊 Traffic metrics, congestion scores and hourly analytics
- 🗺️ Geospatial camera data for map visualisation
- 💾 Dataset collection — images saved for custom model training
    """,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router,      prefix="/api",              tags=["Health"])
app.include_router(cameras.router,     prefix="/api/cameras",      tags=["Cameras"])
app.include_router(detections.router,  prefix="/api/detections",   tags=["Detections"])
app.include_router(metrics.router,     prefix="/api/metrics",      tags=["Metrics"])
app.include_router(annotate.router,    prefix="",                  tags=["Annotate"])


@app.get("/", tags=["Root"])
async def root():
    from app.services.scheduler import scheduler
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "author": "Sumit Malviya",
        "supervisor": "Dr. Jason Moore",
        "university": "Northumbria University",
        "docs": "/api/docs",
        "scheduler": scheduler.status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )