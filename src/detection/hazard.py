# src/detection/hazard.py
# Live, per-frame adaptation of the rule in scripts/hazard_detect.py
# (in the model-training project, not this dashboard repo): is a
# tracked person holding a dangerous object.
#
# Same rule, same reasoning as that script: rule-based, not trained,
# because every class used here (knife, scissors, fork, person) is
# already in COCO, so nothing needs fine-tuning or hand-labelling, and
# an alert like "a knife was detected within 40px of this person's
# wrist across 2 consecutive samples" is auditable in a way a learned
# risk score is not.
#
# Also carried over unchanged: no precision/recall is reported. There
# is no labelled ground truth for "person holding a knife" in this
# project's data, so there is nothing to score against. Treat this as
# a triage signal, not a validated classifier, same as the fall
# detector in state_machine.py, and for the same reason: no labelled
# data for either event type exists anywhere in this project.
#
# Performance note: object detection for knife/scissors/fork does not
# run its own model here. It rides along on PersonDetector's existing
# per-frame YOLO pass (detector.py's `extra_classes` param) instead of
# a second full-frame detector. Class filtering is cheap; the conv
# backbone is the expensive part and it was already running for person
# detection anyway. Only the pose model, a separate network needed for
# wrist keypoints, is owned and throttled by this module.

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
    shared pass only reports the classes that matter (for example,
    skips 'fork' entirely when min_severity='high').
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


def _extract_wrists(pose_result):
    """Pull left/right wrist points out of one Ultralytics pose Result.
    Shared by the single-frame and batched pose paths below."""
    wrists = []
    if pose_result.keypoints is not None and len(pose_result.keypoints.xy):
        for kp in pose_result.keypoints.xy.cpu().numpy():
            for wi in (LEFT_WRIST, RIGHT_WRIST):
                pt = kp[wi]
                if not np.all(pt == 0):   # 0,0 means undetected
                    wrists.append(pt)
    return wrists


# Sentinel key the streak dict is tracked under. See _fire_events'
# docstring for why this is one counter, not one per label.
_STREAK_KEY = "_any_hazard_object"


def _fire_events(hazard_objects, wrists, prox_px, min_consecutive, streak):
    """Pure function: given this frame's already-detected hazard objects,
    this frame's wrist points, the proximity threshold in px, the
    debounce length, and a streak dict to read/mutate, returns newly
    fired events (at most one, since there is only one streak now -- see
    below).

    Extracted out of HazardDetector so the streak/debounce bookkeeping is
    identical whether the pose call that produced `wrists` was made
    per-camera (HazardDetector.check_objects, single-camera/legacy) or
    batched across many cameras in one call (MultiCameraEngine, via
    HazardDetector.predict_wrists_batch). Only where `wrists` came from
    differs, not what happens with it afterward.

    The debounce streak is tracked as a single "some hazard object is
    near a wrist" count (under _STREAK_KEY), not one count per label.
    This used to be per-label, which broke in practice: a
    general-purpose COCO model run at the confidences typical here
    (0.4-0.6, right at the detector's conf floor) can flicker between
    visually similar hazard classes -- knife and scissors both being a
    handheld blade shape -- from one sampled frame to the next, for the
    same physical object. With a per-label streak, "scissors" reaching
    count 1 and then the very next sample reclassifying as "knife"
    starts a *different* counter at 1, and neither ever reaches
    min_consecutive even though a dangerous object was continuously
    near a wrist the whole time. What actually matters for the alert is
    "something dangerous is being held", not "the classifier agreed
    with itself on the exact label twice in a row" -- so the streak
    tracks the former, and reports the label that won the plurality of
    samples across the whole streak -- not just whichever candidate
    happened to be closest to a wrist on the specific sample that
    crossed min_consecutive. Reporting only the trigger sample's label
    had the same flicker problem one level up: a real knife correctly
    read as "knife" on 2 of 3 debounced samples but misread as
    "scissors" on the third (the confusion is exactly what motivated
    tracking one streak instead of one per label, see above) would still
    get logged and alerted as "Scissors Detected" if that third sample
    happened to be the one that hit min_consecutive -- a coin flip
    against the debounce window length, not a reflection of what was
    actually seen most. An exact tie in sample count (common at the
    default min_consecutive=2, where "one each way" has no majority by
    definition) breaks toward whichever label had the higher total
    confidence across its samples.

    `streak` is a plain dict the caller owns and mutates in place.
    check_objects() passes its own instance's self._streak, which is
    fine there since in single-camera/legacy use one HazardDetector
    instance only ever serves one camera. The batched engine passes a
    separate dict per camera, since one room's object-possession streak
    must never bleed into another room's debounce count. The pose model
    is shared for the batched forward pass, but this bookkeeping is not.
    """
    if not wrists or not hazard_objects:
        streak.clear()
        return []

    # Whichever candidate is currently closest to any wrist drives this
    # sample's streak update, and casts this sample's vote toward the
    # fired event's eventual label -- see _votes/_last_seen below.
    best_dist, best_label, best_conf, best_xyxy = min(
        (min(_point_to_box_distance(wpt, xyxy) for wpt in wrists), label, conf, xyxy)
        for label, conf, xyxy in hazard_objects
    )

    if best_dist > prox_px:
        streak.clear()
        return []

    streak[_STREAK_KEY] = streak.get(_STREAK_KEY, 0) + 1

    # Plurality-vote bookkeeping: which label won each sample in the
    # current streak, that label's total summed confidence (the tiebreak
    # below), and that label's most recent (confidence, bbox) so the
    # fired event can report real detection data for whichever label
    # ends up winning, even if that label wasn't this exact sample's
    # winner. Cleared alongside the streak count above/below so a vote
    # tally never survives past its own streak into an unrelated one.
    votes       = streak.setdefault("_label_votes", {})
    conf_totals = streak.setdefault("_label_conf_totals", {})
    last_seen   = streak.setdefault("_label_last_seen", {})
    votes[best_label]       = votes.get(best_label, 0) + 1
    conf_totals[best_label] = conf_totals.get(best_label, 0.0) + best_conf
    last_seen[best_label]   = (best_conf, best_xyxy)

    if streak[_STREAK_KEY] != min_consecutive:
        return []

    # Most samples wins; an exact tie (common at the default
    # min_consecutive=2 -- one sample each way has no majority by
    # definition) breaks toward whichever label the model was more
    # confident about overall, rather than an arbitrary "whichever
    # sample happened to be last" that has nothing to do with which read
    # was more likely correct.
    winning_label = max(votes, key=lambda label: (votes[label], conf_totals[label]))
    winning_conf, winning_xyxy = last_seen[winning_label]
    severity = LABEL_SEVERITY.get(winning_label, "low")
    return [{
        "object": winning_label,
        "severity": severity,
        "detection_conf": round(winning_conf, 3),
        # wrist_distance_px is still this triggering sample's distance
        # (a real, current measurement of "how close right now"), not
        # tied to whichever sample winning_label itself won.
        "wrist_distance_px": round(best_dist, 1),
        "detail": f"wrist within {prox_px:.0f}px across "
                  f"{min_consecutive} consecutive samples",
        # The winning label's own bbox from whichever sample it won, so
        # a caller can draw a box on it (see pipeline.py's
        # _flag_hazard_box) instead of only logging/alerting with no
        # visual confirmation of what was flagged.
        "bbox": winning_xyxy,
    }]


class HazardDetector:
    """
    Wraps a pose model to flag a person holding a dangerous object.
    Takes already-detected hazard objects as input (see module docstring
    for why object detection itself is not owned here) and checks
    whether any of them sit within proximity_frac of a wrist, debounced
    over min_consecutive samples so one noisy frame cannot fire an
    alert.

    See scripts/hazard_detect.py (in the model-training project) for the
    full original design rationale, including why an "unattended
    appliance" rule was deliberately cut: appliance on/off state is not
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
        # Smaller imgsz means faster pose inference, at some cost to
        # keypoint accuracy on small/distant people. None uses the
        # Ultralytics default (640). Tune via HAZARD_IMGSZ in config if
        # this pass is a bottleneck on your hardware.
        self.imgsz = imgsz
        self._streak = {}   # {"_any_hazard_object": consecutive-sample count}, see _fire_events

    def _predict_kwargs(self):
        kwargs = {"verbose": False, "device": self.device}
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        return kwargs

    def predict_wrists(self, frame):
        """Stateless: one pose-model forward pass on one frame, returns
        this frame's wrist points. No streak/debounce logic here; see
        check_objects() (single-camera) and _fire_events() (shared)."""
        pose = self.pose_model.predict(frame, **self._predict_kwargs())[0]
        return _extract_wrists(pose)

    def predict_wrists_batch(self, frames):
        """Stateless and batched: one pose-model forward pass across
        multiple frames (for example, one per camera) instead of one
        call per frame. This is the piece MultiCameraEngine uses so
        hazard pose inference does not scale 1:1 with camera count.
        Returns a list of wrist-point lists, same length and order as
        `frames`.
        """
        results = self.pose_model.predict(list(frames), **self._predict_kwargs())
        return [_extract_wrists(r) for r in results]

    def check_objects(self, frame, hazard_objects):
        """Returns newly fired hazard events given this frame's already
        detected hazard objects. Single-camera/legacy path: this
        instance owns both the pose model and the streak state
        (self._streak) for whichever one camera is calling it.

        hazard_objects: list of (label, conf, xyxy), the extra_detections
        PersonDetector.detect() returned for this frame. Pose only runs
        if at least one candidate object is present, the same gating as
        the original offline script, where a cheap check gates an
        expensive one.
        """
        if not hazard_objects:
            self._streak.clear()
            return []

        h, w = frame.shape[:2]
        prox_px = self.proximity_frac * float(np.hypot(h, w))
        wrists  = self.predict_wrists(frame)
        return _fire_events(hazard_objects, wrists, prox_px, self.min_consecutive, self._streak)
