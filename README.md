# Real-Time Traffic Accident Detection and Automated Emergency Reporting System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-orange.svg)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Ingestion-green.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-ACID%20Storage-003B57.svg)](https://sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, full-stack AI-assisted traffic monitoring, real-time vehicle accident detection, automated emergency reporting, and multi-agency response coordination platform.

---

## System Architecture

```
                                  +-----------------------------+
                                  |   CCTV / RTSP Stream Ingest |
                                  |    (Physical / Synthetic)   |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |     YOLOv8 Object Detection |
                                  |  (Car, Motorcycle, Bus, Trk)|
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |   Collision Heuristic Engine|
                                  |  (IoU Overlap + Confidence) |
                                  +--------------+--------------+
                                                 |
                    +----------------------------+----------------------------+
                    |                                                         |
                    v                                                         v
   +---------------------------------+                       +---------------------------------+
   |   Automated Incident Recording  |                       |  Citizen Mobile Report (/report)|
   |  - Severity Scoring (P1/P2/P3)  |                       |  - Photo Evidence Validation    |
   |  - Nominatim Reverse Geocoding  |                       |  - Geolocation GPS Tagging      |
   +----------------+----------------+                       +----------------+----------------+
                    |                                                         |
                    +----------------------------+----------------------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |   Operator Review Queue     |
                                  |  (Atomic Verification / Rej)|
                                  +--------------+--------------+
                                                 | (If Verified)
                                                 v
                                  +-----------------------------+
                                  |  Emergency SOS Dispatcher   |
                                  | - Haversine Hospital Rank   |
                                  | - Trauma Level 1 Prioritiz. |
                                  | - Hospital Accept / Decline |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |   Admin Analytics & Hotspot |
                                  | - Leaflet Density Heatmap   |
                                  | - Hourly / Monthly Trends   |
                                  | - Actual Response Time KPI  |
                                  +-----------------------------+
```

---

## Role-Based Portals

| Role | Portal URL | Primary Capabilities |
|---|---|---|
| **Administrator** | `/admin` | Complete KPI overview, interactive Leaflet accident hotspot heatmap, hourly/monthly trends, response time analytics, user registration. |
| **Traffic Operator** | `/traffic` | Multi-camera CCTV surveillance, live MJPEG stream monitoring, on-demand YOLO collision inference, traffic congestion evaluation, citizen report verification queue. |
| **EMS Responder** | `/ems` | Real-time hospital trauma center map, 5 km radius Haversine ranking, audible siren alerts, one-click hospital dispatch acceptance/diversion. |
| **Citizen** | `/report` / `/citizen` | Mobile-responsive accident submission form, camera photo upload with magic-byte validation, browser GPS location detection, confidential review queue. |

Default development accounts:
- **Administrator**: `admin@traffic.org` / `AdminPass123!`
- **Traffic Operator**: `officer@traffic.org` / `SecurePass123!`
- **EMS Responder**: `ems@citygov.org` / `ResponderPass123!`
- **Citizen**: `citizen@public.org` / `CitizenPass123!`

---

## Key Features & Jira Sprint Epics

- **SCRUM-22 & SCRUM-37 (Authentication & Role-Based Authorization)**:
  - Persistent SQLite storage for users (`users.db`) with unique indexing.
  - Salted password hashing with `bcrypt`.
  - Signed HS256 JWT access tokens with configurable expiration (default 8 hours).
  - 5-strike consecutive failed login lockout protection.
  - Role-based dependencies (`get_current_user`, `get_current_operator`, `get_current_admin`).

- **SCRUM-39 (Professional Login & Navigation Interface)**:
  - Responsive dark-blue theme (`#0f172a`).
  - Show/hide password visibility toggle.
  - Asynchronous authentication with loading spinner and structured error alerts.
  - Automatic post-login role redirection to appropriate dashboards.

- **SCRUM-40 (Multi-Camera RTSP Video Ingestion Pipeline)**:
  - Multi-threaded `CameraManager` and `RTSPStreamHandler`.
  - Supports physical RTSP URLs, local MP4/AVI video files, webcams, and synthetic test matrices.
  - Real-time stream telemetry (FPS calculation, status monitoring, thread-safe buffering, automatic reconnection).
  - Dedicated camera stream endpoints: `GET /api/cameras`, `GET /api/cameras/{id}/stream`.

- **SCRUM-41 (YOLOv8 Vehicle Detection & Collision Engine)**:
  - Vehicle classification (Car, Motorcycle, Bus, Truck).
  - IoU collision heuristic calculation flagging potential traffic crashes.
  - Graceful fallback when model weights are not present (simulation mode).
  - Synthetic crash generation for demonstration and automated testing.

- **SCRUM-27 (Automated Emergency SOS Dispatch)**:
  - Nearby hospital discovery using Haversine spherical distance calculation.
  - Configurable 5 km coverage radius with trauma facility prioritization for critical crashes.
  - Dispatch record persistence in SQLite with idempotency to prevent duplicate dispatches.
  - Hospital acceptance and decline/diversion endpoints with response time logging.
  - Audible high-priority notification via Web Audio API.

- **SCRUM-34 (Admin Analytics & Interactive Hotspot Heatmap)**:
  - Real-time aggregations from SQLite incident records: Total, Verified, Pending, Rejected, Critical.
  - Hourly accident frequency breakdown (00:00 - 23:00).
  - Severity distribution doughnut chart and monthly incident trends via Chart.js.
  - Interactive Leaflet geographical map with density heat layer (`leaflet-heat`).
  - Filter bar supporting start date, end date, severity level, and verification status.
  - Response time KPI derived from actual stored database timestamps.

- **SCRUM-31 & SCRUM-35 (Traffic Congestion, Rerouting & Citizen Reports)**:
  - Density threshold evaluation (`CONGESTION_VEHICLE_THRESHOLD`).
  - Dynamic routing fallback when external routing API is unconfigured.
  - Citizen photo upload with PIL verification, random UUID storage, and size limits.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- `uv` or `pip` package manager
- Modern web browser (Chrome, Edge, Firefox)

### 2. Clone Repository
```bash
git clone https://github.com/Sividharan/Real-Time_Traffic_Accident_Detection_and_Automated_Emergency_Reporting_System.git
cd Real-Time_Traffic_Accident_Detection_and_Automated_Emergency_Reporting_System
git checkout feature/final-sprint-integration
```

### 3. Create Virtual Environment & Install Dependencies
```bash
# Using standard Python venv:
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r "accident detection/requirements.txt"
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configuration Options:
```env
# Secret key for JWT signing
JWT_SECRET_KEY=traffic-system-super-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=8

# Traffic Congestion Threshold
CONGESTION_VEHICLE_THRESHOLD=5

# Camera Mode (use 'synthetic' for built-in CCTV simulator)
CAMERA_SOURCE=synthetic

# Optional External Routing API (leave empty for built-in fallback)
ROUTING_API_URL=
ROUTING_API_KEY=

# Optional Emergency Dispatch Webhook (leave empty for simulated mode)
EMERGENCY_DISPATCH_API_URL=
EMERGENCY_DISPATCH_API_KEY=

# Model Weight Path
YOLO_MODEL_PATH=yolov8n.pt
```

### 5. Start Application Server
```bash
cd "accident detection/app"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
- **Login Portal**: [http://localhost:8000/login](http://localhost:8000/login)
- **Admin Dashboard**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **Traffic Surveillance**: [http://localhost:8000/traffic](http://localhost:8000/traffic)
- **EMS Response Center**: [http://localhost:8000/ems](http://localhost:8000/ems)
- **Citizen Report Form**: [http://localhost:8000/report](http://localhost:8000/report)
- **Interactive Map**: [http://localhost:8000/map](http://localhost:8000/map)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running Automated Tests

Run the full automated test suite (48 comprehensive tests):

```bash
pytest "accident detection/tests" -v
```

Test coverage includes:
- Authentication & lockout security
- Video stream ingestion and multi-camera handling
- YOLO object detection & collision heuristics
- Emergency hospital search & dispatch workflows
- Analytics aggregations & hotspot calculations
- End-to-end incident lifecycle integration

---

## API Endpoints Reference

### Authentication
- `POST /api/auth/login` - Authenticate credentials and receive JWT
- `POST /api/auth/register` - Create user account (Admin only)
- `GET /api/auth/me` - Get profile of current session
- `POST /api/auth/logout` - Invalidate active session

### Camera Streams & Surveillance
- `GET /api/cameras` - List all cameras and telemetry
- `GET /api/cameras/{id}/status` - Status for specific camera
- `GET /api/cameras/{id}/stream` - MJPEG live stream for specific camera
- `GET /api/stream/live` - Default live camera feed
- `GET /api/system/health` - Health check

### Detection & AI Pipeline
- `GET /api/pipeline/run-inference` - YOLO vehicle detection & collision assessment
- `GET /api/pipeline/visual-detection` - Annotated JPEG frame with bounding boxes
- `GET /api/traffic/congestion` - Vehicle density & congestion broadcast
- `GET /api/traffic/reroute` - Dynamic route calculation

### Citizen Reports & Verification
- `POST /api/incidents` - Submit manual accident report with photo & coordinates
- `GET /api/incidents/pending` - Pending verification queue (Operator only)
- `GET /api/incidents/{id}` - Retrieve incident details
- `POST /api/incidents/{id}/verify` - One-click operator verification & dispatch trigger
- `POST /api/incidents/{id}/reject` - One-click rejection with reason
- `GET /api/incidents/{id}/photo` - Secure image evidence retrieval

### Emergency Dispatch
- `GET /api/emergency/nearby-hospitals` - Ranked hospitals within configurable radius
- `POST /api/emergency/dispatch` - Dispatch accident to hospital
- `GET /api/emergency/dispatch/{incident_id}` - Dispatch status for incident
- `POST /api/emergency/dispatch/{dispatch_id}/accept` - Hospital accepts dispatch
- `POST /api/emergency/dispatch/{dispatch_id}/reject` - Hospital declines dispatch

### Admin Analytics
- `GET /api/analytics/overview` - Incident KPI cards
- `GET /api/analytics/hourly` - Hourly accident counts (00:00-23:00)
- `GET /api/analytics/monthly` - Monthly incident trends
- `GET /api/analytics/severity` - Severity distribution
- `GET /api/analytics/hotspots` - Coordinates and severity for heatmap
- `GET /api/analytics/response-times` - Response time metrics
- `GET /api/analytics/recent-incidents` - Filtered incident logs
