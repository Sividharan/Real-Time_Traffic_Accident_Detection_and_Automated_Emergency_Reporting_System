import threading
import time
import cv2
import numpy as np

class RTSPStreamHandler:
    def __init__(self, camera_id: str, source: str = "synthetic"):
        self.camera_id = camera_id
        self.source = source
        self.status = "Offline"
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self):
        # Convert integer index if digits provided
        src = int(self.source) if str(self.source).isdigit() else self.source

        while self.running:
            if self.source == "synthetic":
                self.status = "Online"
                # Generate a mock 640x480 frame for testing
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"CAM FEED: {self.camera_id}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                with self.lock:
                    self.latest_frame = frame
                time.sleep(0.05)
                continue

            # Standard RTSP / MP4 / Device capture
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.status = "Offline"
                time.sleep(2)
                continue

            self.status = "Online"
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Loop video if testing a file
                    if isinstance(src, str) and not src.startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.status = "Offline"
                    break
                with self.lock:
                    self.latest_frame = frame
                time.sleep(0.03)

            cap.release()
            time.sleep(1)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None