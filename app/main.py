from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import cameras, detections, metrics, health
from app.core.config import settings
from app.db.database import engine, Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables ready")
    print(f"API running on {settings.API_HOST}:{settings.API_PORT}")
    yield


app = FastAPI(
    title="Urban Digital Twin API",
    description="Real-time traffic state estimation using YOLOv8 and Newcastle Urban Observatory cameras.",
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(detections.router, prefix="/api/detections", tags=["Detections"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "author": "Sumit Malviya",
        "university": "Northumbria University",
        "docs": "/api/docs",
    }