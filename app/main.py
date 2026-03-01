"""
Urban Digital Twin - Backend API
Real-Time Traffic State Estimation for Newcastle
Author: Sumit Malviya
Supervisor: Dr. Jason Moore
Northumbria University
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import cameras, detections, metrics, health
from app.core.config import settings
from app.db.database import engine, Base

# Create database tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created")
    print(f"🚀 Urban Digital Twin API started on {settings.API_HOST}:{settings.API_PORT}")
    yield
    # Shutdown
    print("👋 Shutting down Urban Digital Twin API")

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
    ## Urban Digital Twin - Traffic Monitoring API
    
    Real-time traffic state estimation using YOLOv8 and Newcastle Urban Observatory camera feeds.
    
    ### Features:
    - 📷 Camera management and status monitoring
    - 🚗 Real-time vehicle, pedestrian, and cyclist detection
    - 📊 Traffic metrics and analytics
    - 🗺️ Geospatial data for map visualisation
    
    ### Data Sources:
    - Newcastle Urban Observatory API
    - YOLOv8 Object Detection
    """,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(detections.router, prefix="/api/detections", tags=["Detections"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": "Real-Time Traffic State Estimation for Newcastle Digital Twin",
        "author": "Sumit Malviya",
        "supervisor": "Dr. Jason Moore",
        "university": "Northumbria University",
        "docs": "/api/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
