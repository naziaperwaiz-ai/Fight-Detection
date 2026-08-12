# src/detection/detector.py
# YOLO11n person detector — detects and crops persons from frame.
# YOLO26 not available in ultralytics yet; using YOLO11n which is
# the latest available nano model. Swap model path in config when available.

from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.4, device="cpu"):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.device = device
        print(f"[DETECTOR] Loaded {model_path} on {device}")

    def detect(self, frame):
        """
        Returns list of (track_id, bbox, crop) for each person detected.
        bbox = (x1, y1, x2, y2)
        crop = cropped person frame
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.conf,
            verbose=False,
            device=self.device,
            tracker="botsort.yaml"
        )[0]

        detections = []
        if results.boxes is None or results.boxes.id is None:
            return detections

        ids   = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        h, w = frame.shape[:2]
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue  # skip tiny boxes
            crop = frame[y1:y2, x1:x2]
            detections.append((int(track_id), (x1, y1, x2, y2), crop))

        return detections

    def draw(self, frame, detections):
        """Draw bounding boxes on frame."""
        for track_id, (x1, y1, x2, y2), _ in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame