# src/detection/detector.py
# YOLO11n person detector: detects and crops persons from frame.
# YOLO26 is not available in ultralytics yet; using YOLO11n, the latest
# available nano model. Swap model path in config when available.

import time

from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.4, device="cpu"):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.device = device
        # Rate-limits the "dropped person detections, no track ids yet"
        # diagnostic in detect() below so a stretch of consecutive
        # id-less frames (for example right after the tracker resets)
        # logs once, not once per frame.
        self._last_dropped_ids_log = 0.0
        print(f"[DETECTOR] Loaded {model_path} on {device}")

    def detect(self, frame, extra_classes=None):
        """
        Returns list of (track_id, bbox, crop) for each person detected.
        bbox = (x1, y1, x2, y2)
        crop = cropped person frame

        extra_classes, if given, is a dict of {coco_class_id: label} for
        other COCO classes to look for in this same forward pass (for
        example, knife/scissors for hazard detection; see
        detection/hazard.py). Class filtering is a near-free
        post-processing step on an already-running detection pass. The
        conv backbone is the expensive part and runs once regardless of
        how many classes are requested, so this avoids loading and
        running a second full detector model to look for different
        classes in the same frame.

        When extra_classes is given, returns a 2-tuple
        (detections, extra_detections), where extra_detections is a list
        of (label, conf, bbox) for whatever extra classes were found.
        When extra_classes is None (default), returns just `detections`,
        matching the return shape before this parameter existed.
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
        # BoT-SORT may not have an id for every box yet, for example the
        # frame a new track first appears in some tracker states. Track
        # ids are only meaningful for the person class anyway, so a
        # missing id array means "no person detections this frame", not
        # "no detections at all". Hazard objects do not need track ids.
        ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else None

        if ids is None and np.any(all_cls == 0):
            # Previously silent: every person-class box in this frame is
            # dropped below (no track_id to key state-machine history on),
            # which is correct behavior, but with zero visibility it looks
            # identical to "the detector saw nobody" from the outside.
            # Rate-limited to once per 10s so a genuinely id-less stretch
            # doesn't spam the log once per frame.
            now = time.time()
            if now - self._last_dropped_ids_log > 10:
                self._last_dropped_ids_log = now
                print("[DETECTOR] tracker returned no ids this frame; "
                      "dropping person detection(s) that have no track_id "
                      "to attach state to (this message is rate-limited)")

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

    def detect_batch(self, frames, extra_classes=None):
        """
        Stateless, batched person/hazard-object detection across multiple
        frames (for example, one per camera) in a single forward pass.
        This is the piece that makes one model call per cycle, rather
        than one per camera, possible. See detection/multi_camera.py for
        the caller.

        Deliberately does not track (no persist=True, no track_id in the
        output). Ultralytics' built-in tracker is designed for sequential
        frames from one continuous stream, and its behavior when fed a
        fresh batch of independent camera frames each cycle is not
        documented as safe; a track_id bleeding across cameras would
        corrupt that person's state-machine history. Track identity for
        a batched multi-camera run is assigned separately, per camera,
        by SimpleIOUTracker in multi_camera.py: cheap CPU bookkeeping,
        not a model, so doing it once per camera does not undermine the
        batching this method exists for.

        Returns a list, same length and order as `frames`. Each element
        is either:
          - detections (extra_classes=None): list of (bbox, crop)
          - (detections, extra_detections) otherwise, extra_detections
            being a list of (label, conf, bbox)
        i.e. detect()'s per-frame return shape, minus the track_id (since
        there is no tracker here to produce one).
        """
        classes = [0] + (list(extra_classes.keys()) if extra_classes is not None else [])
        results = self.model.predict(
            list(frames),
            classes=classes,
            conf=self.conf,
            verbose=False,
            device=self.device,
        )

        out = []
        for frame, r in zip(frames, results):
            detections       = []
            extra_detections = []
            if r.boxes is not None:
                h, w      = frame.shape[:2]
                all_boxes = r.boxes.xyxy.cpu().numpy()
                all_cls   = r.boxes.cls.cpu().numpy().astype(int)
                all_conf  = r.boxes.conf.cpu().numpy()
                for i in range(len(all_boxes)):
                    cls = int(all_cls[i])
                    x1, y1, x2, y2 = all_boxes[i].astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if cls == 0:
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue  # skip tiny boxes, same as detect()
                        crop = frame[y1:y2, x1:x2]
                        detections.append(((x1, y1, x2, y2), crop))
                    elif extra_classes is not None and cls in extra_classes:
                        extra_detections.append((extra_classes[cls], float(all_conf[i]), (x1, y1, x2, y2)))
            out.append((detections, extra_detections) if extra_classes is not None else detections)
        return out

    def draw(self, frame, detections):
        """Draw bounding boxes on frame."""
        for track_id, (x1, y1, x2, y2), _ in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame