# 🚦 Urban Digital Twin - Backend API

Real-Time Traffic State Estimation using Deep Learning and Live Camera Feeds for Newcastle Digital Twin.

**Author:** Sumit Malviya  
**Supervisor:** Dr. Jason Moore  
**University:** Northumbria University  
**Module:** KF7029 MSc Computer Science Project

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Database](#database)
- [YOLO Detection](#yolo-detection)
- [Testing](#testing)

---

## 🎯 Overview

This backend API powers the Urban Digital Twin dashboard, providing:
- Real-time traffic data from Newcastle Urban Observatory cameras
- YOLOv8-based object detection for vehicles, pedestrians, and cyclists
- RESTful API endpoints for the React frontend
- Historical data storage and analytics

---

## ✨ Features

- **📷 Camera Management** - Track and manage traffic camera feeds
- **🤖 YOLO Detection** - Automated object detection using YOLOv8
- **📊 Traffic Metrics** - Real-time and historical traffic analytics
- **🗺️ Geospatial Data** - Location-based camera and traffic data
- **⚡ Async API** - Fast, non-blocking FastAPI endpoints
- **🐘 PostgreSQL** - Robust data storage with async support
- **🐳 Docker Ready** - Easy deployment with Docker Compose

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI |
| **Database** | PostgreSQL / SQLite |
| **ORM** | SQLAlchemy (async) |
| **Detection** | YOLOv8 (Ultralytics) |
| **Data Source** | Newcastle Urban Observatory API |
| **Validation** | Pydantic |
| **Server** | Uvicorn |

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── cameras.py   # Camera endpoints
│   │   │   ├── detections.py # Detection endpoints
│   │   │   ├── metrics.py   # Analytics endpoints
│   │   │   └── health.py    # Health check endpoints
│   │   └── __init__.py
│   ├── core/
│   │   ├── config.py        # Application settings
│   │   └── __init__.py
│   ├── db/
│   │   ├── database.py      # Database connection
│   │   └── __init__.py
│   ├── models/
│   │   ├── models.py        # SQLAlchemy models
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── __init__.py
│   └── services/
│       ├── yolo_detector.py      # YOLO detection service
│       ├── urban_observatory.py  # Urban Observatory collector
│       ├── traffic_calculator.py # Traffic metrics calculator
│       └── __init__.py
├── data/
│   └── images/              # Stored camera images
├── tests/
│   └── __init__.py
├── .env.example             # Environment variables template
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker build configuration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- PostgreSQL (optional, SQLite for development)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sumitCoderNotFound/urban-digital-twin.git
cd urban-digital-twin/backend
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

### Step 5: Download YOLO Model

```bash
# The model will auto-download on first use, or:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## ▶️ Running the API

### Development Mode (with auto-reload)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## 📖 API Documentation

Once running, access the interactive API documentation:

- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc
- **OpenAPI JSON:** http://localhost:8000/api/openapi.json

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cameras` | List all cameras |
| GET | `/api/cameras/{id}` | Get camera details |
| GET | `/api/cameras/{id}/detections` | Get camera detections |
| GET | `/api/detections` | List detections |
| POST | `/api/detections` | Create detection |
| GET | `/api/metrics/totals` | Get total metrics |
| GET | `/api/metrics/hourly` | Get hourly data |
| GET | `/api/health` | Health check |

---

## ⚙️ Configuration

Configuration is managed through environment variables. See `.env.example` for all options.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `USE_SQLITE` | Use SQLite instead of PostgreSQL | `true` |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `YOLO_MODEL` | YOLO model file | `yolov8n.pt` |
| `YOLO_CONFIDENCE` | Detection confidence threshold | `0.5` |
| `COLLECTION_INTERVAL` | Image collection interval (seconds) | `300` |

---

## 🐘 Database

### Using SQLite (Development)

SQLite is used by default for easy development:

```env
USE_SQLITE=true
```

### Using PostgreSQL (Production)

```env
USE_SQLITE=false
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/urban_twin
```

### Database Models

- **Camera** - Traffic camera locations and metadata
- **Detection** - YOLO detection results
- **HourlyMetric** - Aggregated hourly statistics
- **SystemLog** - System logs and events

---

## 🤖 YOLO Detection

### Supported Classes

| Class ID | Name | Category |
|----------|------|----------|
| 0 | person | Pedestrian |
| 1 | bicycle | Cyclist |
| 2 | car | Vehicle |
| 3 | motorcycle | Vehicle |
| 5 | bus | Vehicle |
| 7 | truck | Vehicle |

### Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `yolov8n.pt` | ⚡ Fastest | Good | Development, CPU |
| `yolov8s.pt` | Fast | Better | Balanced |
| `yolov8m.pt` | Medium | Good | Production |
| `yolov8l.pt` | Slow | Great | High accuracy |
| `yolov8x.pt` | Slowest | Best | Maximum accuracy |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_cameras.py
```

---

## 📝 License

This project is part of an MSc dissertation at Northumbria University.

---

## 🙏 Acknowledgements

- **Newcastle Urban Observatory** for providing traffic camera data
- **Ultralytics** for YOLOv8
- **Dr. Jason Moore** for supervision and guidance
# urban-traffic-digital-twin-backend
