import os
from typing import Any, Dict, List, Optional
import numpy as np


class AccidentDetector:
    """
    Vehicle and potential accident detection engine using YOLOv8 with collision heuristics.
    Detects cars, motorcycles, buses, and trucks.
    Evaluates spatial IoU overlap between bounding boxes to flag collision events.
    Gracefully falls back to simulated development mode when weights are missing.
    """

    def __init__(self, model_weight: str = "yolov8n.pt", conf_thresh: float = 0.85):
        self.conf_thresh = conf_thresh
        self.target_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.model = None
        self.model_available = False
        self.weight_path = model_weight

        # Try resolving relative path inside app or models dir
        potential_paths = [
            model_weight,
            os.path.join(os.path.dirname(__file__), model_weight),
            os.path.join(os.path.dirname(__file__), "..", "models", model_weight),
            os.path.join(os.path.dirname(__file__), "..", "..", model_weight),
        ]

        resolved_path = None
        for p in potential_paths:
            if os.path.isfile(p):
                resolved_path = p
                break

        if resolved_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(resolved_path)
                self.model_available = True
                self.weight_path = resolved_path
            except Exception as e:
                self.model_available = False
                self.load_error = str(e)
        else:
            # Try loading default weights or mark simulation mode
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_weight)
                self.model_available = True
            except Exception as e:
                self.model_available = False
                self.load_error = str(e)

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculates Intersection-over-Union between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = float(area1 + area2 - intersection + 1e-6)
        return intersection / union

    def evaluate_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs object detection and collision evaluation on an input OpenCV BGR frame.
        """
        if not self.model_available or self.model is None or frame is None:
            # Fallback when model is unavailable or frame is blank
            return {
                "accident_detected": False,
                "crash_boxes": [],
                "vehicle_count": 0,
                "detected_vehicles": [],
                "simulated": True,
                "model_available": self.model_available,
                "note": "Potential accident detection heuristic (simulation mode)"
            }

        try:
            results = self.model(frame, verbose=False, conf=0.4)[0]
            vehicles = []
            detected_vehicles = []
            accident_detected = False
            crash_boxes = []

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id in self.target_classes:
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    label = self.target_classes[cls_id]
                    vehicles.append((coords, conf, label))
                    detected_vehicles.append({
                        "class": label,
                        "confidence": round(conf, 3),
                        "box": [round(c, 1) for c in coords]
                    })

            # Collision heuristic: Overlapping vehicle bounding boxes with IoU > 0.30
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    box_a, conf_a, label_a = vehicles[i]
                    box_b, conf_b, label_b = vehicles[j]
                    iou = self.calculate_iou(box_a, box_b)
                    avg_confidence = (conf_a + conf_b) / 2.0

                    if iou > 0.30 and avg_confidence >= self.conf_thresh:
                        accident_detected = True
                        crash_boxes.extend([box_a, box_b])

            return {
                "accident_detected": accident_detected,
                "crash_boxes": crash_boxes,
                "vehicle_count": len(vehicles),
                "detected_vehicles": detected_vehicles,
                "simulated": False,
                "model_available": True,
                "note": "Potential accident detection based on bounding box collision heuristic"
            }
        except Exception:
            return {
                "accident_detected": False,
                "crash_boxes": [],
                "vehicle_count": 0,
                "detected_vehicles": [],
                "simulated": True,
                "model_available": False,
                "note": "Inference exception handled safely"
            }

    def simulate_accident_event(self) -> Dict[str, Any]:
        """Provides reproducible simulated accident detection telemetry for tests & demonstrations."""
        return {
            "accident_detected": True,
            "crash_boxes": [[120, 150, 280, 320], [230, 160, 390, 330]],
            "vehicle_count": 2,
            "simulated": True,
            "confidence": 0.895,
            "detected_vehicles": [
                {"class": "car", "confidence": 0.91, "box": [120, 150, 280, 320]},
                {"class": "truck", "confidence": 0.88, "box": [230, 160, 390, 330]}
            ],
            "severity_estimate": "Critical",
            "note": "Simulated collision event for development testing"
        }
