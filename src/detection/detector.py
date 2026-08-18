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

    def detect(self, frame, extra_classes=None):
        """
        Returns list of (track_id, bbox, crop) for each person detected.
        bbox = (x1, y1, x2, y2)
        crop = cropped person frame

        extra_classes, if given, is a dict of {coco_class_id: label} for
        OTHER COCO classes to look for in this SAME forward pass (e.g.
        knife/scissors for hazard detection -- see detection/hazard.py).
        Class filtering is a near-free post-processing step on an
        already-running detection pass -- the expensive part is the conv
        backbone, which runs once regardless of how many classes you ask
        for -- so this avoids loading and running a second full detector
        model just to look for different classes in the same frame.

        When extra_classes is given, returns a 2-tuple
        (detections, extra_detections), where extra_detections is a list
        of (label, conf, bbox) for whatever extra classes were found.
        When extra_classes is None (default), returns just `detections`,
        exactly as before -- existing callers are unaffected.
        """
        classes = [0] + (list(extra_classes.keys()) if extra_classes is not None else [])
        results = self.model.track(
            frame,
            persist=True,
            classes=classes,
            conf=self.conf,
            verbose=False,
            device=self.device,
            tracker="botsort.yaml"
        )[0]

        detections       = []
        extra_detections = []

        if results.boxes is None:
            return (detections, extra_detections) if extra_classes is not None else detections

        h, w = frame.shape[:2]
        all_boxes = results.boxes.xyxy.cpu().numpy()
        all_cls   = results.boxes.cls.cpu().numpy().astype(int)
        all_conf  = results.boxes.conf.cpu().numpy()
        # BoT-SORT may not have an id for every box yet (e.g. the frame
        # a new track first appears in some tracker states) -- track ids
        # are only meaningful for the person class anyway, so a missing
        # id array just means "no person detections this frame", not
        # "no detections at all". Hazard objects don't need track ids.
        ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else None

        for i in range(len(all_boxes)):
            cls = int(all_cls[i])
            x1, y1, x2, y2 = all_boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if cls == 0:
                if ids is None:
                    continue
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue  # skip tiny boxes
                crop = frame[y1:y2, x1:x2]
                detections.append((int(ids[i]), (x1, y1, x2, y2), crop))
            elif extra_classes is not None and cls in extra_classes:
                extra_detections.append((extra_classes[cls], float(all_conf[i]), (x1, y1, x2, y2)))

        return (detections, extra_detections) if extra_classes is not None else detections

    def draw(self, frame, detections):
        """Draw bounding boxes on frame."""
        for track_id, (x1, y1, x2, y2), _ in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame