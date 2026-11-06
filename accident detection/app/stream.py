import os
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


class RTSPStreamHandler:
    """
    Robust RTSP / Video Stream ingestion handler.
    Supports physical RTSP streams, local video files, webcams, and synthetic test feeds.
    Features thread-safe frame access, reconnection logic, and health metrics.
    """

    def __init__(self, camera_id: str, source: str = "synthetic", name: Optional[str] = None):
        self.camera_id = camera_id
        self.source = source
        self.name = name or camera_id
        self.status = "Initializing"
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.last_frame_time = 0.0
        self.reconnect_attempts = 0
        self.fps = 0.0
        self._frame_count = 0
        self._last_fps_calc = time.time()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.status = "Offline"

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates dynamic simulated traffic CCTV frame for development and demonstration."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Asphalt roadway
        cv2.rectangle(frame, (80, 0), (560, 480), (45, 45, 45), -1)
        # Lane markings (dashed)
        t = int(time.time() * 2) % 2
        for y in range(20, 480, 60):
            cv2.line(frame, (320, y + t * 10), (320, y + 30 + t * 10), (200, 200, 200), 3)

        # Overhead telemetry overlay
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"FEED: {self.name} [{self.camera_id}]", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
        cv2.putText(frame, f"TIME: {timestamp} (SYNTHETIC CCTV)", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

        # Simulated traffic vehicles moving
        offset = int((time.time() * 40) % 360)
        cv2.rectangle(frame, (160, 100 + offset), (220, 180 + offset), (0, 180, 0), -1)
        cv2.putText(frame, "CAR", (165, 95 + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.rectangle(frame, (380, 300 - offset), (460, 420 - offset), (180, 100, 0), -1)
        cv2.putText(frame, "BUS", (385, 295 - offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame

    def _capture_loop(self) -> None:
        src = int(self.source) if str(self.source).isdigit() else self.source

        while self.running:
            if str(self.source).lower() == "synthetic":
                self.status = "Online"
                frame = self._generate_synthetic_frame()
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                self._update_fps()
                time.sleep(0.04)  # ~25 FPS
                continue

            # Physical RTSP, Device, or Video File capture
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.status = "Offline"
                self.reconnect_attempts += 1
                time.sleep(min(self.reconnect_attempts * 1.5, 10.0))
                continue

            self.status = "Online"
            self.reconnect_attempts = 0

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video if testing a local file
                    if isinstance(src, str) and not src.startswith("rtsp://"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.status = "Offline"
                    break

                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_time = time.time()
                self._update_fps()
                time.sleep(0.03)

            cap.release()
            time.sleep(1.0)

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_calc
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_fps_calc = now

    def get_frame(self) -> Optional[np.ndarray]:
        """Thread-safe copy of latest ingested frame."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_status_dict(self) -> Dict[str, Any]:
        """Detailed health and telemetry dictionary for this camera feed."""
        now = time.time()
        is_healthy = (self.status == "Online") and ((now - self.last_frame_time) < 5.0)
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": "synthetic" if self.source == "synthetic" else "rtsp/file",
            "status": "Online" if is_healthy else "Offline",
            "stream_healthy": is_healthy,
            "fps": self.fps if is_healthy else 0.0,
            "last_frame_age_seconds": round(now - self.last_frame_time, 2) if self.last_frame_time > 0 else None,
            "reconnect_attempts": self.reconnect_attempts
        }


class CameraManager:
    """Manages multiple camera stream handlers simultaneously with thread safety."""

    def __init__(self):
        self.streams: Dict[str, RTSPStreamHandler] = {}
        self._lock = threading.Lock()

    def add_camera(self, camera_id: str, source: str = "synthetic", name: Optional[str] = None) -> RTSPStreamHandler:
        with self._lock:
            if camera_id in self.streams:
                # If already exists, return existing
                return self.streams[camera_id]
            handler = RTSPStreamHandler(camera_id=camera_id, source=source, name=name)
            handler.start()
            self.streams[camera_id] = handler
            return handler

    def get_stream(self, camera_id: str) -> Optional[RTSPStreamHandler]:
        with self._lock:
            return self.streams.get(camera_id)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        stream = self.get_stream(camera_id)
        return stream.get_frame() if stream else None

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [stream.get_status_dict() for stream in self.streams.values()]

    def get_camera_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        stream = self.get_stream(camera_id)
        return stream.get_status_dict() if stream else None

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            stream = self.streams.pop(camera_id, None)
            if stream:
                stream.stop()
                return True
            return False

    def shutdown_all(self) -> None:
        with self._lock:
            for stream in self.streams.values():
                stream.stop()
            self.streams.clear()
