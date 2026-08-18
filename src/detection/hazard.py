# src/detection/hazard.py
# Live, per-frame adaptation of the rule in scripts/hazard_detect.py
# (in the model-training project, not this dashboard repo): is a
# tracked person holding a dangerous object.
#
# Same rule, same reasoning as that script -- rule-based, not trained,
# because every class used here (knife, scissors, fork, person) is
# already in COCO, so nothing needs fine-tuning or hand-labelling, and
# an alert like "a knife was detected within 40px of this person's
# wrist across 2 consecutive samples" is auditable in a way a learned
# risk score is not.
#
# Also carried over unchanged: no precision/recall is reported. There
# is no labelled ground truth for "person holding a knife" in this
# project's data, so there is nothing to score against. Treat this as
# a triage signal, not a validated classifier -- same as the fall
# detector in state_machine.py, and for the same reason (no labelled
# data for either event type exists anywhere in this project).
#
# PERFORMANCE NOTE: object detection for knife/scissors/fork does NOT
# run its own model here. It rides along on PersonDetector's existing
# per-frame YOLO pass (detector.py's `extra_classes` param) instead of
# a second full-frame detector -- class filtering is cheap, the conv
# backbone is the expensive part and it was already running for person
# detection anyway. Only the pose model (a genuinely different network,
# needed for wrist keypoints) is owned and throttled by this module.

import numpy as np
from ultralytics import YOLO

# COCO class ids for handheld objects where possession alone is the hazard.
# Severity is attached per object: a knife and a fork should not raise the
# same alarm, or the mealtime false-positive rate makes the whole feed
# unreadable. This dict is the single source of truth for both what
# detector.py is asked to look for (hazard_class_map) and the severity
# attached to whatever fires (LABEL_SEVERITY).
HAZARD_CLASSES = {
    43: ("knife", "high"),
    76: ("scissors", "high"),
    42: ("fork", "low"),
}
LABEL_SEVERITY = {label: sev for label, sev in HAZARD_CLASSES.values()}
SEVERITY_RANK  = {"low": 0, "high": 1}

# COCO keypoint indices for wrists.
LEFT_WRIST, RIGHT_WRIST = 9, 10


def hazard_class_map(min_severity="high"):
    """{coco_class_id: label}, filtered to severities >= min_severity.

    Passed to PersonDetector.detect(frame, extra_classes=...) so the
    shared pass only bothers reporting the classes we actually care
    about (e.g. skip 'fork' entirely when min_severity='high').
    """
    min_rank = SEVERITY_RANK[min_severity]
    return {
        cid: label for cid, (label, sev) in HAZARD_CLASSES.items()
        if SEVERITY_RANK[sev] >= min_rank
    }


def _point_to_box_distance(point, xyxy):
    """Shortest distance from a point to a box, 0 if inside.

    Distance to the box rather than to its centre matters for elongated
    objects: a hand at the handle of a long knife is touching it, even
    though the box centre may be far away.
    """
    x1, y1, x2, y2 = xyxy
    dx = max(x1 - point[0], 0, point[0] - x2)
    dy = max(y1 - point[1], 0, point[1] - y2)
    return float(np.hypot(dx, dy))


class HazardDetector:
    """
    Wraps a pose model to flag a person holding a dangerous object.
    Takes already-detected hazard objects as input (see module docstring
    for why object detection itself isn't owned here) and checks whether
    any of them sit within proximity_frac of a wrist, debounced over
    min_consecutive samples so one noisy frame can't fire an alert.

    See scripts/hazard_detect.py (in the model-training project) for the
    full original design rationale, including why an "unattended
    appliance" rule was deliberately cut -- appliance on/off state isn't
    recoverable from a standard RGB camera.

    Weights auto-download from Ultralytics on first use if not already
    cached locally.
    """

    def __init__(self, pose_weights="yolov8n-pose.pt", device="cpu",
                 proximity_frac=0.06, min_consecutive=2, imgsz=None):
        self.pose_model = YOLO(pose_weights)
        self.device     = device
        self.proximity_frac  = proximity_frac
        self.min_consecutive = min_consecutive
        # Smaller imgsz = faster pose inference, at some cost to keypoint
        # accuracy on small/distant people. None = ultralytics default
        # (640). Tune via HAZARD_IMGSZ in config if this pass is a
        # bottleneck on your hardware.
        self.imgsz = imgsz
        self._streak = {}   # object label -> consecutive-sample count

    def check_objects(self, frame, hazard_objects):
        """Returns newly-fired hazard events given this frame's already-
        detected hazard objects.

        hazard_objects: list of (label, conf, xyxy) -- the extra_detections
        PersonDetector.detect() returned for this frame. Pose only runs
        if at least one candidate object is present, same gating as the
        original offline script (a cheap check gates an expensive one).
        """
        if not hazard_objects:
            self._streak.clear()
            return []

        h, w = frame.shape[:2]
        prox_px = self.proximity_frac * float(np.hypot(h, w))

        predict_kwargs = {"verbose": False, "device": self.device}
        if self.imgsz:
            predict_kwargs["imgsz"] = self.imgsz
        pose = self.pose_model.predict(frame, **predict_kwargs)[0]

        wrists = []
        if pose.keypoints is not None and len(pose.keypoints.xy):
            for kp in pose.keypoints.xy.cpu().numpy():
                for wi in (LEFT_WRIST, RIGHT_WRIST):
                    pt = kp[wi]
                    if not np.all(pt == 0):   # 0,0 means undetected
                        wrists.append(pt)
        if not wrists:
            self._streak.clear()
            return []

        fired = []
        seen_labels = set()
        for label, conf, xyxy in hazard_objects:
            seen_labels.add(label)
            severity = LABEL_SEVERITY.get(label, "low")
            dist = min(_point_to_box_distance(wpt, xyxy) for wpt in wrists)
            if dist <= prox_px:
                self._streak[label] = self._streak.get(label, 0) + 1
                if self._streak[label] == self.min_consecutive:
                    fired.append({
                        "object": label,
                        "severity": severity,
                        "detection_conf": round(conf, 3),
                        "wrist_distance_px": round(dist, 1),
                        "detail": f"wrist within {prox_px:.0f}px across "
                                  f"{self.min_consecutive} consecutive samples",
                    })
            else:
                self._streak[label] = 0

        # A label that simply wasn't detected this sample (object briefly
        # out of view, a missed frame) must not leave its old streak count
        # sitting there to be resumed later -- "consecutive" should mean
        # consecutive, not "seen min_consecutive times total, with gaps".
        for label in list(self._streak):
            if label not in seen_labels:
                self._streak[label] = 0

        return fired
