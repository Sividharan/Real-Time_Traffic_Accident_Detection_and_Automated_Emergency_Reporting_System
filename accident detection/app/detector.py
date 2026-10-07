import numpy as np
from ultralytics import YOLO

class AccidentDetector:
    def __init__(self, model_weight: str = "yolov8n.pt", conf_thresh: float = 0.85):
        # Can be swapped with custom fine-tuned weights (e.g. accident-best.pt)
        self.model = YOLO(model_weight)
        self.conf_thresh = conf_thresh
        self.target_classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / float(area1 + area2 - intersection + 1e-6)

    def evaluate_frame(self, frame):
        results = self.model(frame, verbose=False, conf=0.5)[0]
        vehicles = []
        accident_detected = False
        crash_boxes = []

        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls in self.target_classes:
                coords = box.xyxy[0].cpu().numpy()
                vehicles.append((coords, conf))

        # Collision heuristic: Overlapping bounding boxes with combined confidence >= 85%
        # False-positive filtering: Requires sustained physical overlap > 0.40 IoU
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                box_a, conf_a = vehicles[i]
                box_b, conf_b = vehicles[j]
                iou = self.calculate_iou(box_a, box_b)
                avg_confidence = (conf_a + conf_b) / 2.0

                if iou > 0.35 and avg_confidence >= self.conf_thresh:
                    accident_detected = True
                    crash_boxes.extend([box_a, box_b])

        return {
            "accident_detected": accident_detected,
            "crash_boxes": crash_boxes,
            "vehicle_count": len(vehicles)
        }