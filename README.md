# Real-Time Traffic Accident Detection and Automated Emergency Reporting System

A comprehensive FastAPI-based solution for real-time traffic monitoring, vehicle accident detection using YOLOv8, automated emergency incident reporting, reverse geocoding via OpenStreetMap, and dynamic traffic congestion alert broadcasting & rerouting (SCRUM-31).

---

## Key Features

- **Real-Time Video Ingestion**: RTSP stream ingestion and synthetic video feed simulation.
- **Accident Detection (YOLOv8)**: Real-time object detection (cars, buses, trucks, motorcycles) with bounding box overlap & IoU collision heuristics.
- **Automated Incident Reporting**: Automatic location payload generation with reverse geocoding via OpenStreetMap Nominatim API.
- **Role-Based Authentication**: Secure JWT-based authentication with bcrypt hashing and 5-strike account lockout protection.
- **Dynamic Traffic Congestion & Rerouting (SCRUM-31)**:
  - **Congestion Detection**: Configurable vehicle density thresholding (`CONGESTION_VEHICLE_THRESHOLD`) to evaluate traffic density without false alarms.
  - **Dynamic Alternative Routing**: Integration engine supporting external routing APIs (e.g., OSRM) with fallback route advisory when unconfigured or unreachable.
  - **Congestion Alert Broadcast**: Broadcasts congestion status, affected location address, vehicle count, and alternative routing details.

---

## Installation Instructions

### Prerequisites
- Python 3.10+ (Python 3.14 compatible)
- `pip` package manager

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Sividharan/Real-Time_Traffic_Accident_Detection_and_Automated_Emergency_Reporting_System.git
   cd Real-Time_Traffic_Accident_Detection_and_Automated_Emergency_Reporting_System
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r "accident detection/requirements.txt" pytest httpx
   ```

---

## Environment Variable Configuration

Create a `.env` file in the project root or `accident detection/` directory based on `.env.example`:

```env
# Traffic Congestion Threshold (Default: 5 vehicles)
CONGESTION_VEHICLE_THRESHOLD=5

# Optional External Routing API URL & Key
ROUTING_API_URL=
ROUTING_API_KEY=

# Secret key for JWT auth
JWT_SECRET_KEY=traffic-system-super-secret-key
```

### External Routing API Setup
If an external OSRM or custom routing service is available, specify `ROUTING_API_URL`:
- Example OSRM Endpoint: `http://router.project-osrm.org/route/v1/driving/`
If unconfigured or if the API service is unreachable, the system automatically provides a safe fallback route advisory.

---

## Application Startup Commands

To start the FastAPI web application with Uvicorn:

```bash
cd "accident detection/app"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive API documentation (Swagger UI) at:
`http://localhost:8000/docs`

---

## Testing Commands

Run the automated test suite covering accident detection, authentication, reverse geocoding, congestion thresholding, routing fallback, and API failures:

```bash
pytest "accident detection/tests" -v
```

---

## API Endpoints Overview

| Endpoint | Method | Description |
|---|---|---|
| `/login` | GET | HTML Login Page |
| `/api/auth/login` | POST | Authenticates traffic operator / responder credentials |
| `/api/system/health` | GET | System health & stream status check |
| `/api/pipeline/run-inference` | GET | Runs YOLO inference, accident evaluation, and congestion check |
| `/api/traffic/congestion` | GET | **[SCRUM-31]** Evaluates traffic density and broadcasts congestion alert |
| `/api/traffic/reroute` | GET | **[SCRUM-31]** Calculates dynamic alternative routes or fallback route |
| `/api/stream/live` | GET | MJPEG live video stream feed |
| `/api/geo/reverse-lookup` | GET | Resolves camera coordinates to street address and landmark |

---

## Known Limitations

- **Routing API Fallback**: When `ROUTING_API_URL` is not set, the system uses static advisory routes and does not fabricate fake travel time savings.
- **Synthetic Feed**: Default camera feed uses a synthetic OpenCV matrix for testing when hardware RTSP cameras are offline.
