# src/detection/pipeline.py
import requests
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from collections import deque
from pathlib import Path
import time
import numpy as np
import threading
import os
import shutil
import subprocess

from detection.config import Config
from notification.notifier import Notifier, load_alert_settings

# cv2.VideoWriter's "mp4v" fourcc (MPEG-4 Part 2 / Simple Profile) writes a
# perfectly valid, fully-playable-in-VLC .mp4 -- but no mainstream browser
# (Chrome, Edge, Firefox, Safari) ships an MPEG-4 Part 2 decoder for
# <video>. A clip written with just mp4v loads in the dashboard player as a
# black frame stuck at 0:00 with a dead scrubber, even though the file on
# disk is complete and the right duration -- that's a codec-support
# failure, not a recording failure. _save_clip below writes the raw frames
# with mp4v (the only fourcc every OpenCV/FFmpeg build can be counted on to
# actually support) and then shells out to the system `ffmpeg` binary to
# transcode that into H.264/yuv420p, which every browser can decode.
# Resolved once at import time rather than per clip.
_FFMPEG_BIN = shutil.which("ffmpeg")


class SimpleIOUTracker:
    """Minimal per-camera IoU-based tracker: greedily matches this frame's
    boxes to last frame's tracked boxes by IoU, assigns a new id to any
    unmatched box, and drops a track once it's gone unmatched for more
    than max_age consecutive calls.

    Built for the batched multi-camera path (see multi_camera.py and
    detector.py's detect_batch docstring). Ultralytics' built-in tracker
    assumes sequential frames from one continuous stream, and its
    behavior on a fresh batch of independent camera frames each cycle is
    undocumented. This tracker is intentionally small, has no model in
    it, and is cheap enough to run once per camera even though the
    detection/classification calls it feeds are batched.

    Not a replacement for BoT-SORT in general: no motion model, no
    re-identification after a long occlusion, no appearance embedding.
    For this project's purpose, attaching a stable-enough id to a person
    for a few seconds so the state machine and violence score history
    stay attributable to the same track, that is sufficient; this does
    not attempt general multi-object tracking.
    """

    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self._next_id      = 1
        self._tracks       = {}   # track_id -> {"bbox": (x1,y1,x2,y2), "age": int}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter  = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, boxes):
        """boxes: list of (x1,y1,x2,y2) for the current frame.
        Returns a list of track_ids, same length and order as boxes."""
        candidates = []
        for bi, b in enumerate(boxes):
            for tid, t in self._tracks.items():
                iou = self._iou(b, t["bbox"])
                if iou >= self.iou_threshold:
                    candidates.append((iou, bi, tid))
        candidates.sort(key=lambda c: c[0], reverse=True)

        assigned      = [None] * len(boxes)
        used_boxes    = set()
        used_track_ids = set()
        for iou, bi, tid in candidates:
            if bi in used_boxes or tid in used_track_ids:
                continue
            assigned[bi] = tid
            used_boxes.add(bi)
            used_track_ids.add(tid)
            self._tracks[tid]["bbox"] = boxes[bi]
            self._tracks[tid]["age"]  = 0

        for bi, b in enumerate(boxes):
            if assigned[bi] is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": b, "age": 0}
                assigned[bi] = tid
                used_track_ids.add(tid)

        for tid in list(self._tracks):
            if tid not in used_track_ids:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        return assigned


class ViolenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)


def _build_violence_transform():
    """The preprocessing pipeline fed to ViolenceClassifier. Factored out
    to a module function so every CameraWorker in shared mode can reuse
    one instance (via MultiCameraEngine's `shared` dict) instead of each
    constructing its own functionally-identical T.Compose -- the object
    is stateless and holds no per-camera data, so there is nothing
    correctness-relevant about giving each worker its own copy, only
    wasted allocation."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CameraWorker:
    """
    Owns one camera. Runs detection in background thread.
    Dashboard reads latest annotated frame via get_frame().
    """
    def __init__(self, config: Config, shared=None):
        """
        shared: optional dict handed in by MultiCameraEngine
            ({"model": ViolenceClassifier, "detector": PersonDetector,
              "hazard_detector": HazardDetector or None,
              "hazard_class_map": dict or None}). When given, this worker
            does not load its own copies of the model, detector, or
            hazard model; it reuses the engine's shared instances, so N
            cameras do not mean N copies of the same weights in memory.
            This worker still owns everything else about being a camera:
            its own StateMachine, alert/recording state, motion baseline,
            and (in this mode) its own SimpleIOUTracker, since
            Ultralytics' persist=True tracker is not safe to share across
            independent camera streams batched through one call. See
            SimpleIOUTracker's docstring and detector.py's detect_batch.

            When None (default), this worker is fully self-contained,
            as it was before the multi-camera refactor: it loads and
            owns its own model, detector, and hazard detector, and its
            run() loop drives itself end to end. This is the path
            single-camera deployments and any direct CameraWorker use
            still take.
        """
        self.cfg    = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = None   # only used in shared mode; see below

        from detection.detector import PersonDetector
        from detection.state_machine import StateMachine

        if shared is not None:
            self.model            = shared["model"]
            self.detector         = shared["detector"]
            self.hazard_detector  = shared.get("hazard_detector")
            self._hazard_class_map = shared.get("hazard_class_map")
            self.tracker           = SimpleIOUTracker()
        else:
            # model
            self.model = ViolenceClassifier()
            # weights_only=True restricts unpickling to tensors and plain
            # data, not arbitrary Python objects. A .pt file is a pickle
            # archive by default, and loading one with weights_only=False
            # (the old default) lets a malicious model file execute
            # arbitrary code the moment it is loaded. Model uploads are
            # admin-only in this app, but a trusted admin can still be
            # handed a bad file by someone else, so this fails loudly
            # rather than falling back to the unsafe path on older torch
            # versions.
            try:
                state_dict = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=True)
            except TypeError:
                raise RuntimeError(
                    "This torch version does not support weights_only=True. "
                    "Upgrade torch (>=1.13) before loading model weights. "
                    "weights_only=False is a code-execution risk."
                )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"[{config.CAMERA_ID}] Model loaded on {self.device}")

            self.detector = PersonDetector(device="cpu")

            # Hazard detection (dangerous-object-near-wrist rule) is
            # opt-in. Object detection for it (knife/scissors/fork) rides
            # along on PersonDetector's existing per-frame pass (see
            # detector.py's extra_classes param) instead of a second full
            # detector model, since class filtering is near-free on an
            # already-running forward pass. Only the pose model is a
            # separate network, so it is the only one still owned and
            # throttled here. See detection/hazard.py.
            self.hazard_detector  = None
            self._hazard_class_map = None
            if getattr(config, "HAZARD_DETECTION_ENABLED", False):
                from detection.hazard import HazardDetector, hazard_class_map
                min_severity = getattr(config, "HAZARD_MIN_SEVERITY", "high")
                self._hazard_class_map = hazard_class_map(min_severity)
                self.hazard_detector = HazardDetector(
                    pose_weights=getattr(config, "HAZARD_POSE_WEIGHTS", "yolov8n-pose.pt"),
                    device="cpu",
                    proximity_frac=getattr(config, "HAZARD_PROXIMITY_FRAC", 0.06),
                    min_consecutive=getattr(config, "HAZARD_MIN_CONSECUTIVE", 2),
                    imgsz=getattr(config, "HAZARD_IMGSZ", 320),
                )
                print(f"[{config.CAMERA_ID}] Hazard detection enabled "
                      f"(min_severity={min_severity}, classes={self._hazard_class_map}, "
                      f"imgsz={getattr(config, 'HAZARD_IMGSZ', 320)})")

        self.state_machine = StateMachine(config)
        self._hazard_frame_count = 0

        self.transform = (shared.get("transform") if shared is not None else None) \
            or _build_violence_transform()

        self.buffer        = deque(maxlen=config.BUFFER_SECONDS * config.FPS)
        # Real wall-clock append time for each frame in self.buffer, kept
        # in lockstep with it (same maxlen, appended alongside every
        # self.buffer.append() call -- see multi_camera.py's _run_cycle
        # and this class's own run()). Exists purely so _save_clip can
        # write a clip's real, achieved frame rate instead of assuming
        # config.FPS was actually achieved -- see _start_recording and
        # _save_clip's docstrings for why that assumption was wrong.
        self.buffer_times  = deque(maxlen=config.BUFFER_SECONDS * config.FPS)
        self.violence_scores = deque(maxlen=config.CONFIRM_SECONDS * config.FPS)
        self.alert_active  = False
        self.last_alert_time = 0
        self.recording     = False
        self.record_frames = []
        self.record_start_time = None
        # Real elapsed seconds the pre-event portion of the current/last
        # recording's frames (self.buffer's contents at the moment
        # _start_recording ran) actually spans, computed from
        # buffer_times rather than assumed from BUFFER_SECONDS -- see
        # _start_recording.
        self._record_pre_event_span = 0.0
        self.notifier      = Notifier(config)
        self._prev_gray    = None

        # Live-feed privacy blur: the frame published for the dashboard
        # (_set_frame, read by get_frame/video_feed) stays a frozen,
        # heavily blurred placeholder except while something is actually
        # happening. self._last_unblur_trigger_time is None until the
        # first trigger, so a camera that has never seen a trigger stays
        # blurred by default rather than starting unblurred. This is
        # entirely separate from self.buffer/self.record_frames above,
        # which keep receiving real, unblurred frames regardless of blur
        # state; a saved alert clip is only ever produced by a real
        # trigger already having fired, so it is meant to show the real
        # footage.
        self._last_unblur_trigger_time = None
        self._live_blur_hysteresis = getattr(config, "LIVE_BLUR_HYSTERESIS_SECONDS", 15)
        self._live_blur_refresh    = getattr(config, "LIVE_BLUR_REFRESH_SECONDS", 10)
        self._blurred_frame        = None
        self._blurred_frame_at     = 0.0

        # Visual confirmation for a fired hazard event: draw a box on the
        # flagged object itself, not just a console line/incident record.
        # Each entry is (bbox, label, expires_at); populated in
        # process_frame() when a hazard event fires (see _flag_hazard_box),
        # drawn in _draw_hazard_boxes(), and pruned once expired. The
        # object detector only re-locates hazard objects on throttled
        # samples (HAZARD_SAMPLE_EVERY_N_FRAMES), so the box is the last
        # known position held for a short window, the same "frozen until
        # the next sample" tradeoff the privacy-blur placeholder already
        # makes, not a live per-frame tracked box.
        self._flagged_hazard_boxes = []
        self._hazard_box_display_seconds = getattr(config, "HAZARD_BOX_DISPLAY_SECONDS", 5)

        # shared frame for dashboard
        self._frame      = None
        self._frame_lock = threading.Lock()
        self.score       = 0.0

    def get_frame(self):
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def _set_frame(self, frame):
        with self._frame_lock:
            self._frame = frame.copy()

    def _motion_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._prev_gray = gray
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(magnitude.mean())

    def _predict(self, frame, detections):
        """Legacy single-camera scoring path: owns the model call itself.
        Only correct to use when this worker is not in shared mode (see
        __init__). In shared mode, scores come from MultiCameraEngine's
        batched classifier call instead, via process_frame()'s
        scores_by_tid argument."""
        if self._motion_score(frame) < self.cfg.MOTION_THRESHOLD:
            return 0.0

        if not detections:
            return 0.0

        scores = []
        for track_id, bbox, crop in detections:
            with torch.no_grad():
                img  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inp  = self.transform(img).unsqueeze(0).to(self.device)
                out  = self.model(inp)
                score = torch.softmax(out, dim=1)[0][1].item()
                scores.append(score)
                self.state_machine.update(track_id, score, bbox, frame)

        return float(max(scores)) if scores else 0.0

    # States that count as "something is actually happening" for the live
    # feed's privacy blur. Proximate is deliberately excluded: two
    # people's boxes being near each other happens constantly during
    # ordinary caregiving contact (helping someone stand, adjusting
    # bedding), and unblurring on that would defeat the point.
    _LIVE_BLUR_TRIGGER_STATES = frozenset({"Agitated", "Fighting", "OnGround", "Emergency"})

    def _live_feed_should_be_unblurred_now(self, hazard_events):
        """Whether this specific frame is itself a trigger: any tracked
        person is at or above Agitated, a fall is currently confirmed, or
        a hazard event fired this frame. Does not consider the hysteresis
        window; see process_frame's caller for that."""
        if any(t.state in self._LIVE_BLUR_TRIGGER_STATES for t in self.state_machine.tracks.values()):
            return True
        if self.state_machine.has_fall():
            return True
        if hazard_events:
            return True
        return False

    def _privacy_blur(self, frame):
        """Downscale-then-upscale pixelation rather than a Gaussian blur:
        cheap, and it destroys detail (faces, skin, objects) far more
        reliably than a blur radius that might still leave a recognizable
        shape at high resolution."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _flag_hazard_box(self, ev):
        """Records that a hazard event just fired, so _draw_hazard_boxes
        below draws a box on it for a short window. ev["bbox"] is the
        flagged object's own bounding box at the moment the streak
        crossed min_consecutive (see hazard.py's _fire_events); older
        code paths / tests that construct an event dict without "bbox"
        are handled by skipping the flag rather than raising, since
        drawing a box is a visual nicety, not something that should be
        able to break alert dispatch above it."""
        bbox = ev.get("bbox")
        if bbox is None:
            return
        expires_at = time.time() + self._hazard_box_display_seconds
        self._flagged_hazard_boxes.append((bbox, ev["object"], expires_at))

    def _draw_hazard_boxes(self, frame):
        """Draws a box + label on every still-live flagged hazard object,
        pruning expired ones first. Distinct color/thickness from
        detector.draw()'s person boxes so a hazard flag reads as visually
        different from ordinary person tracking, not just another box in
        the same style."""
        now = time.time()
        self._flagged_hazard_boxes = [
            (bbox, label, expires_at) for bbox, label, expires_at in self._flagged_hazard_boxes
            if expires_at > now
        ]
        for (x1, y1, x2, y2), label, _ in self._flagged_hazard_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(frame, f"HAZARD: {label}", (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def _start_recording(self, now):
        """Begins buffering a post-event clip: snapshots the pre-event
        buffer (self.buffer already holds the last BUFFER_SECONDS worth
        of raw frames), marks recording active, and starts the
        POST_EVENT_SECONDS countdown that process_frame's own recording
        block below uses to know when to hand the clip off to
        _save_clip(). Shared by every alert path that should produce a
        clip -- the violence-score-threshold path and hazard events, as
        of this method existing -- so "what starts a recording" has one
        implementation, not one copy-pasted per alert type. self.recording
        is used as the buffering flag; self.alert_active additionally
        gates *starting a new* recording (see both call sites), so an
        already-in-progress clip is never interrupted or its buffer
        reset by a second event firing mid-recording, regardless of
        which alert type started it or which fires next.

        Also snapshots how much real wall-clock time the pre-event
        buffer frames actually span, into self._record_pre_event_span,
        for _save_clip to use -- see that method's docstring for why
        assuming config.FPS was actually achieved produced clips whose
        reported duration didn't match how long the buffer/recording
        window actually was.
        """
        self.alert_active      = True
        self.recording         = True
        self.record_frames     = list(self.buffer)
        self.record_start_time = now
        self._record_pre_event_span = (
            (now - self.buffer_times[0]) if self.buffer_times else 0.0
        )

    def _publish_live_frame(self, frame, hazard_events):
        """Decides what the dashboard's live feed actually gets for this
        frame, and publishes it via _set_frame. Does not affect the
        return value of process_frame() (still the real annotated
        frame), so the debug tool's cv2.imshow window (which uses that
        return value directly, not get_frame()) is unaffected. See the
        blur-state fields set in __init__ for what each one tracks."""
        now = time.time()
        if self._live_feed_should_be_unblurred_now(hazard_events):
            self._last_unblur_trigger_time = now

        within_hysteresis = (
            self._last_unblur_trigger_time is not None and
            (now - self._last_unblur_trigger_time) <= self._live_blur_hysteresis
        )

        if within_hysteresis:
            self._set_frame(frame)
            return

        # Frozen, with a periodic refresh so a caregiver can tell the
        # camera is still alive rather than staring at a placeholder that
        # never changes, without that refresh being a live blurred video
        # (which would still leak motion/shape through the blur).
        if self._blurred_frame is None or (now - self._blurred_frame_at) >= self._live_blur_refresh:
            self._blurred_frame    = self._privacy_blur(frame)
            self._blurred_frame_at = now
        self._set_frame(self._blurred_frame)

    def _check_hazard(self, frame, hazard_objects):
        """Legacy single-camera hazard pose check: calls the pose model
        itself, throttled by HAZARD_SAMPLE_EVERY_N_FRAMES. Returns the
        list of fired hazard events (possibly empty), same shape as
        HazardDetector.check_objects(). Not used in shared mode;
        MultiCameraEngine batches the pose call across cameras and passes
        the resulting events straight into process_frame().

        Gating is on the sample throttle alone, not on hazard_objects
        being present: a due sample with no candidate object still needs
        to reach check_objects() so its streak gets cleared (see
        check_objects()'s own `if not hazard_objects` branch). Skipping
        the call entirely on an empty due sample would let a stale
        streak from an earlier sighting survive a real gap and resume
        counting on the next detection instead of restarting at 1."""
        self._hazard_frame_count += 1
        sample_every = max(1, getattr(self.cfg, "HAZARD_SAMPLE_EVERY_N_FRAMES", 5))
        if self._hazard_frame_count % sample_every != 0:
            return []
        try:
            return self.hazard_detector.check_objects(frame, hazard_objects)
        except Exception as e:
            # A transient inference error (a malformed frame, a
            # first-run weight download hiccup) should not take down the
            # whole camera thread. Skip this sample and keep going,
            # matching the alert POSTs elsewhere that already do this for
            # network failures.
            print(f"[{self.cfg.CAMERA_ID}] Hazard check failed, skipping sample: {e}")
            return []

    def run(self):
        """Dev/debug only, not part of the production path. This worker
        owns its own model and detector and drives itself end to end,
        showing the annotated feed in a cv2.imshow window for one camera
        at a time. main.py never calls this; production always goes
        through MultiCameraEngine (see multi_camera.py), even for a
        single camera, since a batch of size 1 there costs the same as
        calling the model directly.

        Kept as a quick way to visually check one camera's detections
        without starting the dashboard. See scripts/debug_single_camera.py
        for the supported way to invoke it.

        Not used when this worker was built in shared mode (see
        __init__): self.detector.detect() here uses Ultralytics'
        persist=True tracker, which is only valid for one continuous
        stream, and a shared detector instance driven this way from
        multiple cameras would corrupt track ids across them.
        MultiCameraEngine drives shared-mode workers itself, via
        process_frame(), and never calls this method."""
        source = int(self.cfg.CAMERA_SOURCE) if str(self.cfg.CAMERA_SOURCE).isdigit() else self.cfg.CAMERA_SOURCE
        cap    = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[{self.cfg.CAMERA_ID}] Cannot open camera: {self.cfg.CAMERA_SOURCE}")
            return

        print(f"[{self.cfg.CAMERA_ID}] Camera opened. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.buffer.append(frame.copy())
            self.buffer_times.append(time.time())
            # When hazard detection is on, knife/scissors/fork classes
            # ride along on this same pass (extra_classes) instead of a
            # second full-frame detector; see detector.py. Falls back to
            # the plain single-list return when hazard is off, so the
            # hot path is unchanged for anyone not using this feature.
            if self.hazard_detector is not None:
                detections, hazard_objects = self.detector.detect(frame, extra_classes=self._hazard_class_map)
            else:
                detections, hazard_objects = self.detector.detect(frame), []

            motion_ok = self._motion_score(frame) >= self.cfg.MOTION_THRESHOLD
            scores_by_tid = {}
            if motion_ok and detections:
                for track_id, bbox, crop in detections:
                    with torch.no_grad():
                        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        inp = self.transform(img).unsqueeze(0).to(self.device)
                        out = self.model(inp)
                        scores_by_tid[track_id] = torch.softmax(out, dim=1)[0][1].item()

            hazard_events = (
                self._check_hazard(frame, hazard_objects)
                if self.hazard_detector is not None else []
            )

            frame = self.process_frame(frame, detections, motion_ok, scores_by_tid, hazard_events)

            cv2.imshow(f"Violence Detection - {self.cfg.CAMERA_ID}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _dispatch_alert(self, notifier_args, api_payload):
        """Sends the alert email and the /api/events/add POST for one
        event off the calling thread.

        Both calls are real network I/O (a blocking SMTP send in
        Notifier.send_alert, an HTTP POST) that can take real time. In
        the single-camera legacy run() loop that only stalled that one
        camera. In the batched MultiCameraEngine, every camera's
        process_frame() call happens on the same shared inference
        thread, so without this, one camera's slow email send would
        stall detection for every other camera too, undermining the
        reason batching exists. Dispatching here keeps "fire the alert"
        from blocking "keep detecting", in both modes.

        Errors are caught and logged rather than raised, matching the
        try/except that already wrapped the POST before this existed.
        A notification failure should never crash anything, foreground
        or background.
        """
        def _send():
            try:
                self.notifier.send_alert(*notifier_args)
            except Exception as e:
                print(f"[{self.cfg.CAMERA_ID}] Alert email failed: {e}")
            try:
                # requests defaults to verifying the server cert against
                # the public CA bundle, which a self-signed cert can
                # never pass -- so once DASHBOARD_URL is https://, this
                # POST silently failed its SSL handshake on every single
                # call (caught by the except below with nothing printed),
                # meaning alerts fired and printed to the console but
                # never once reached Incident History. DASHBOARD_CERT_PATH
                # is the exact cert file the dashboard itself serves (see
                # main.py's _ssl_context/__main__), so `verify` trusts
                # precisely that known cert instead of either failing
                # outright or disabling verification wholesale. None
                # (falls back to True) when serving plain HTTP, where
                # verify has no effect on a non-https URL anyway.
                requests.post(
                    f"{self.cfg.DASHBOARD_URL}/api/events/add",
                    json=api_payload,
                    headers={"X-Internal-Key": getattr(self.cfg, "INTERNAL_API_KEY", "")},
                    timeout=1.5,
                    verify=getattr(self.cfg, "DASHBOARD_CERT_PATH", None) or True,
                )
            except Exception as e:
                # Previously silent (a bare `except Exception: pass`),
                # which is exactly why the DASHBOARD_URL scheme mismatch
                # and this SSL-verification issue both went undetected
                # for so long: alerts appeared to work (printed to the
                # console, email attempted) while silently never reaching
                # Incident History. A failure here should still never
                # crash detection, but it must not be invisible either.
                print(f"[{self.cfg.CAMERA_ID}] Incident history POST failed: {e}")
        threading.Thread(target=_send, daemon=True).start()

    def process_frame(self, frame, detections, motion_ok, scores_by_tid=None, hazard_events=None):
        """Everything that happens to one already-detected-and-scored
        frame: state-machine update, hazard-event alerting, violence
        alerting, clip recording, and drawing. Factored out of run()'s
        loop body so the batched multi-camera engine (which computes
        detections and scores across all cameras via shared calls) and
        the legacy single-camera run() loop (which computes them itself,
        one camera at a time) drive the same downstream behavior. There
        is one implementation of what an alert means, not two that can
        drift apart.

        detections: list of (track_id, bbox, crop) for this frame, from
            detector.detect() (legacy) or a batch entry from
            detector.detect_batch() plus a per-camera tracker (engine).
        motion_ok: whether this frame cleared MOTION_THRESHOLD. Mirrors
            the original _predict()'s early return: when False, no
            violence score is computed and the state machine's per-track
            update() is skipped entirely for this frame, rather than
            updated with score 0.0, which would dilute avg_score()
            differently than never having been called.
        scores_by_tid: {track_id: violence_score}, consulted only when
            motion_ok. A detection with no entry is treated as 0.0.
        hazard_events: already-fired hazard events for this frame (see
            HazardDetector.check_objects), or None/[] if hazard detection
            is off or was not due this frame. The pose-model call itself
            is made by the caller, not here; batching it across cameras
            is the reason this split exists.

        Returns the annotated frame.
        """
        scores_by_tid = scores_by_tid or {}
        hazard_events = hazard_events or []

        # Read once per frame rather than once per cooldown check below,
        # so a caregiver's saved Alert Settings cooldown actually gates
        # real alerts (previously only test_alert() read this file at
        # all), without four redundant file reads per frame. Falls back
        # to the Config default when the settings file is missing,
        # unreadable, or has no cooldown key yet.
        cooldown_seconds = load_alert_settings().get("cooldown", self.cfg.COOLDOWN_SECONDS)

        score = 0.0
        if motion_ok and detections:
            for track_id, bbox, crop in detections:
                s = scores_by_tid.get(track_id, 0.0)
                score = max(score, s)
                self.state_machine.update(track_id, s, bbox, frame)
        self.state_machine.update_all(detections)

        for ev in hazard_events:
            # Flagging the box is deliberately outside the cooldown gate
            # below: cooldown exists to stop repeat emails/incident spam
            # for the same ongoing hazard, not to hide visual state a
            # caregiver looking at the live feed right now should still
            # see. An event only reaches hazard_events at all once
            # per streak crossing min_consecutive (see hazard.py's
            # _fire_events), so this cannot fire on every frame either
            # way.
            self._flag_hazard_box(ev)
            now = time.time()
            # Recording is likewise started outside the cooldown gate,
            # same reasoning as the box flag above: a caregiver should
            # get a saved clip of a hazard event even if the alert email/
            # incident-log entry itself is being suppressed by the
            # cooldown because a different alert fired moments ago.
            # Guarded on alert_active (not cooldown) so an already
            # in-progress recording -- from this hazard event or from a
            # violence alert -- is never interrupted or its buffer reset.
            if not self.alert_active:
                self._start_recording(now)
                print(f"[{self.cfg.CAMERA_ID}] Recording clip for hazard event: {ev['object']}")
            if (now - self.last_alert_time) > cooldown_seconds:
                self.last_alert_time = now
                label = f"Hazard Detected: {ev['object']}"
                print(f"[{self.cfg.CAMERA_ID}] {label} "
                      f"(conf={ev['detection_conf']}, {ev['detail']})")
                self._dispatch_alert(
                    (self.cfg.CAMERA_ID, self.cfg.ROOM_NAME, label, ev["detection_conf"], "Saving..."),
                    {
                        "camera_id":  self.cfg.CAMERA_ID,
                        "room":       self.cfg.ROOM_NAME,
                        "event_type": "Hazard Detected",
                        "confidence": ev["detection_conf"],
                        "clip_path":  "Saving...",
                        "detail": (f"{ev['object']} ({ev['severity']} severity), "
                                   f"{ev['detail']}"),
                    },
                )

        frame      = self.detector.draw(frame, detections)
        frame      = self._draw_hazard_boxes(frame)
        self.score = score
        self.violence_scores.append(score)

        avg_score  = float(np.mean(self.violence_scores))
        confirmed  = (
            avg_score >= self.cfg.VIOLENCE_THRESHOLD and
            len(self.violence_scores) == self.violence_scores.maxlen
        )
        now        = time.time()
        cooldown_ok = (now - self.last_alert_time) > cooldown_seconds

        if confirmed and cooldown_ok and not self.alert_active:
            self._start_recording(now)
            self.last_alert_time   = now
            print(f"[{self.cfg.CAMERA_ID}] ALERT! Score: {avg_score:.2f}")

            # Snapshot each currently-tracked person's state and score so
            # the dashboard's incident detail view can show real "people
            # tracked" data. This is available in state_machine at the
            # moment the alert fires.
            states_snapshot = [
                {"track_id": tid, "state": track.state, "score": track.avg_score()}
                for tid, track in self.state_machine.tracks.items()
            ]
            self._dispatch_alert(
                (self.cfg.CAMERA_ID, self.cfg.ROOM_NAME, "Violence Detected", avg_score, "Saving..."),
                {
                    "camera_id":  self.cfg.CAMERA_ID,
                    "room":       self.cfg.ROOM_NAME,
                    "event_type": "Violence Detected",
                    "confidence": avg_score,
                    "clip_path":  "Saving...",
                    "states":     states_snapshot,
                },
            )

        if self.recording:
            self.record_frames.append(frame.copy())
            if now - self.record_start_time >= self.cfg.POST_EVENT_SECONDS:
                # Snapshot the reference before resetting
                # self.record_frames below. `self.record_frames = []`
                # rebinds the attribute to a new empty list rather than
                # clearing the existing one in place, so the object
                # frames_to_save points at stays intact and untouched by
                # that reset, and is safe to hand to a background thread.
                # _save_clip() takes it as an explicit argument rather
                # than reading self.record_frames itself because, by the
                # time a background thread actually runs, self.record_frames
                # may already be the fresh empty list, not the frames
                # that need saving.
                frames_to_save = self.record_frames
                # Real wall-clock span the whole clip covers: the
                # pre-event buffer's own measured span (snapshotted in
                # _start_recording) plus however long the post-event
                # portion actually took in real time (now -
                # record_start_time -- close to POST_EVENT_SECONDS by
                # design, but not assumed exact). Handed to _save_clip so
                # it can write the clip at its real achieved frame rate
                # instead of assuming config.FPS was actually achieved --
                # see _save_clip's docstring.
                elapsed_seconds = self._record_pre_event_span + (now - self.record_start_time)
                threading.Thread(
                    target=self._save_clip, args=(frames_to_save, elapsed_seconds), daemon=True
                ).start()
                self.recording    = False
                self.alert_active = False
                self.record_frames = []

        # annotate frame
        color = (0, 0, 255) if avg_score >= self.cfg.VIOLENCE_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Violence: {avg_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if self.alert_active:
            cv2.putText(frame, "ALERT", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # draw state for each person
        states = self.state_machine.get_states()
        for tid, state in states.items():
            track = self.state_machine.tracks.get(tid)
            if track and track.bbox_history:
                x1, y1, x2, y2 = track.bbox_history[-1]
                state_color = {
                    "Normal":    (0, 255, 0),
                    "Proximate": (0, 255, 255),
                    "Agitated":  (0, 165, 255),
                    "Fighting":  (0, 0, 255),
                    "OnGround":  (255, 0, 255),
                    "Emergency": (255, 0, 0),
                }.get(state, (255,255,255))
                cv2.putText(frame, state, (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        # emergency escalation
        if self.state_machine.has_emergency():
            now = time.time()
            # Recording is started outside the cooldown gate below, same
            # reasoning as the violence and hazard paths above: a
            # caregiver should get a saved clip of an emergency even if
            # the alert email/incident-log entry itself is being
            # suppressed by the cooldown because a different alert fired
            # moments ago. Guarded on alert_active so an already
            # in-progress recording is never interrupted or reset.
            if not self.alert_active:
                self._start_recording(now)
                print(f"[{self.cfg.CAMERA_ID}] Recording clip for emergency event")
            if (now - self.last_alert_time) > cooldown_seconds:
                self.last_alert_time = now
                # Emergency events used to only send an email and never
                # reached the dashboard's incident history, so they were
                # invisible in the UI. Logged the same way as the other
                # alert types so it shows up in Incident history/Analytics.
                emergency_states = [
                    {"track_id": tid, "state": track.state, "score": track.avg_score()}
                    for tid, track in self.state_machine.tracks.items()
                ]
                self._dispatch_alert(
                    (self.cfg.CAMERA_ID, self.cfg.ROOM_NAME, "EMERGENCY - Person Down 30s", 1.0, "Saving..."),
                    {
                        "camera_id":  self.cfg.CAMERA_ID,
                        "room":       self.cfg.ROOM_NAME,
                        "event_type": "Emergency",
                        "confidence": 1.0,
                        "clip_path":  "Saving...",
                        "states":     emergency_states,
                    },
                )

        # fall escalation, independent of the violence pipeline above.
        # See state_machine.py's _update_fall for the rule and its
        # caveats. Shares the same cooldown clock as the other alert
        # types deliberately: one alerting budget per camera, not a
        # separate one per event type.
        if self.state_machine.has_fall():
            now = time.time()
            # Same recording pattern as the emergency block above: start
            # outside the cooldown gate, guarded only on alert_active, so
            # a fall gets a saved clip even when the alert itself is
            # cooldown-suppressed.
            if not self.alert_active:
                self._start_recording(now)
                print(f"[{self.cfg.CAMERA_ID}] Recording clip for fall event")
            if (now - self.last_alert_time) > cooldown_seconds:
                self.last_alert_time = now
                print(f"[{self.cfg.CAMERA_ID}] FALL DETECTED (geometry rule, not a model score)")
                fall_states = [
                    {"track_id": tid, "state": track.state,
                     "fall_status": track.fall_status, "score": track.avg_score()}
                    for tid, track in self.state_machine.tracks.items()
                ]
                self._dispatch_alert(
                    (self.cfg.CAMERA_ID, self.cfg.ROOM_NAME, "Fall Detected", 1.0, "Saving..."),
                    {
                        "camera_id":  self.cfg.CAMERA_ID,
                        "room":       self.cfg.ROOM_NAME,
                        "event_type": "Fall Detected",
                        # 1.0 here means the geometry rule fired, not a
                        # model probability, matching the convention used
                        # for the Emergency event above. It should not be
                        # read as equivalent to the violence classifier's
                        # score.
                        "confidence": 1.0,
                        "clip_path":  "Saving...",
                        "states":     fall_states,
                    },
                )

        # Fighting escalation, independent of the score-threshold alert
        # above. has_fighting() is a state-machine-level signal (the
        # "Fighting" state, set from proximity/agitation/motion, not the
        # violence classifier score directly) that previously had no
        # caller anywhere in the codebase; without this block, a camera
        # could sit in the Fighting state indefinitely with no alert if
        # avg_score never crossed VIOLENCE_THRESHOLD. Shares the same
        # cooldown clock as every other alert type on this camera
        # deliberately: one alerting budget per camera, not a separate
        # one per event type.
        if self.state_machine.has_fighting():
            now = time.time()
            # Same recording pattern as the emergency/fall blocks above:
            # start outside the cooldown gate, guarded only on
            # alert_active, so a fighting event gets a saved clip even
            # when the alert itself is cooldown-suppressed.
            if not self.alert_active:
                self._start_recording(now)
                print(f"[{self.cfg.CAMERA_ID}] Recording clip for fighting event")
            if (now - self.last_alert_time) > cooldown_seconds:
                self.last_alert_time = now
                # motion_confirmed_fight is set by
                # StateMachine._update_motion_fight_pair when a track
                # reached Fighting via the motion/proximity backup
                # signal rather than avg_score() crossing
                # VIOLENCE_THRESHOLD. Surfacing it here tells an operator
                # (or anyone reading logs later) *why* this alert fired
                # even if the printed/stored confidence looks low -- the
                # classifier didn't confirm it, sustained mutual motion
                # did.
                motion_confirmed = any(
                    getattr(track, "motion_confirmed_fight", False)
                    for track in self.state_machine.tracks.values()
                )
                if motion_confirmed:
                    print(f"[{self.cfg.CAMERA_ID}] FIGHTING DETECTED (motion/proximity backup signal, "
                          f"classifier score was {avg_score:.2f})")
                else:
                    print(f"[{self.cfg.CAMERA_ID}] FIGHTING DETECTED (state machine)")
                fighting_states = [
                    {"track_id": tid, "state": track.state, "score": track.avg_score(),
                     "motion_confirmed_fight": getattr(track, "motion_confirmed_fight", False)}
                    for tid, track in self.state_machine.tracks.items()
                ]
                self._dispatch_alert(
                    (self.cfg.CAMERA_ID, self.cfg.ROOM_NAME, "Fighting Detected", avg_score, "Saving..."),
                    {
                        "camera_id":  self.cfg.CAMERA_ID,
                        "room":       self.cfg.ROOM_NAME,
                        "event_type": "Fighting Detected",
                        "confidence": avg_score,
                        "clip_path":  "Saving...",
                        "states":     fighting_states,
                        "motion_confirmed_fight": motion_confirmed,
                    },
                )

        # draw fall status per person, separate from the violence-state
        # label already drawn above, since a person can be Normal
        # violence-wise and mid-fall at the same time
        for tid, track in self.state_machine.tracks.items():
            if track.fall_status != "None" and track.bbox_history:
                x1, y1, x2, y2 = track.bbox_history[-1]
                fall_color = (0, 140, 255) if track.fall_status == "Suspected" else (0, 0, 255)
                cv2.putText(frame, f"FALL:{track.fall_status}", (x1, max(0, y1 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, fall_color, 2)

        self._publish_live_frame(frame, hazard_events)
        return frame

    def _save_clip(self, frames=None, elapsed_seconds=None):
        """Writes the buffered alert clip to disk and notifies.

        frames: explicit snapshot of the frames to save. Callers dispatch
        this method onto a background thread (see process_frame) so a
        slow disk write does not stall the shared multi-camera inference
        loop, and so writing a clip for camera A never blocks detection
        on cameras B through T. Callers must pass frames explicitly
        rather than let this read self.record_frames itself, because by
        the time a background thread actually runs, the caller may
        already have reset self.record_frames to a new empty list.
        Defaults to self.record_frames for any direct or synchronous
        caller (for example, tests) that is not going through that
        dispatch path.

        elapsed_seconds: real wall-clock time the pre-event buffer plus
        post-event recording window actually spanned (see process_frame's
        recording-finalize block and _start_recording). Used to write the
        clip at the frame rate it was actually captured at, instead of
        assuming config.FPS -- the camera's *configured* target rate --
        was actually achieved. Under MultiCameraEngine's batched cycle,
        the real per-camera capture/append rate can fall well short of
        config.FPS under load (more cameras, hazard pose detection
        enabled, a slow host); writing far fewer frames than
        config.FPS implies at config.FPS produces a technically valid
        .mp4 whose reported duration (frame_count / fps) is a small
        fraction of how long the recording window actually ran -- for a
        short enough frame count, that rounds to "0:00" in most players,
        even though a real multi-second event was captured. Falls back to
        config.FPS when elapsed_seconds is missing or too small to trust
        (a direct/synchronous caller not going through process_frame, or
        a clock edge case), matching the previous behavior exactly in
        that case.
        """
        frames = self.record_frames if frames is None else frames
        if not frames:
            return
        Path(self.cfg.CLIPS_DIR).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path  = (Path(self.cfg.CLIPS_DIR) / f"alert_{self.cfg.CAMERA_ID}_{timestamp}.mp4").as_posix()
        # Raw mp4v write always lands here first; see _FFMPEG_BIN's comment
        # above for why this alone is not the browser-playable deliverable.
        raw_path  = (Path(self.cfg.CLIPS_DIR) / f"_raw_{self.cfg.CAMERA_ID}_{timestamp}.mp4").as_posix()
        h, w      = frames[0].shape[:2]
        # A fraction-of-a-second elapsed_seconds (clock hiccup, or a
        # direct caller that never went through _start_recording) can't
        # be trusted to divide by -- fall back to the configured FPS
        # rather than producing an absurd (or infinite/zero-division)
        # writer FPS.
        if elapsed_seconds and elapsed_seconds >= 1.0:
            fps = max(1.0, len(frames) / elapsed_seconds)
        else:
            fps = self.cfg.FPS
        writer    = cv2.VideoWriter(
            raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )
        for f in frames:
            writer.write(f)
        writer.release()

        # Transcode the mp4v intermediate into H.264 so the dashboard's
        # <video> player can actually decode it. Skipped (rather than
        # crashing _save_clip) when the raw file never materialized --
        # e.g. tests that monkeypatch cv2.VideoWriter with a fake that
        # doesn't touch disk -- or when ffmpeg isn't installed, in which
        # case the raw mp4v file is kept under out_path as a best-effort
        # fallback: it won't play in-browser, but it's still a real,
        # complete recording that VLC/ffplay/etc. can open, rather than
        # being silently discarded.
        if not Path(raw_path).exists():
            pass
        elif _FFMPEG_BIN is None:
            print(f"[{self.cfg.CAMERA_ID}] ffmpeg not found on PATH -- "
                  f"keeping raw mp4v clip (will not play in the dashboard's "
                  f"browser player): {out_path}")
            os.replace(raw_path, out_path)
        else:
            try:
                subprocess.run(
                    [_FFMPEG_BIN, "-y", "-loglevel", "error",
                     "-i", raw_path,
                     "-c:v", "libx264", "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     out_path],
                    check=True, capture_output=True, timeout=60,
                )
                os.remove(raw_path)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                detail = getattr(e, "stderr", b"")
                detail = detail.decode(errors="replace") if isinstance(detail, bytes) else detail
                print(f"[{self.cfg.CAMERA_ID}] ffmpeg transcode failed "
                      f"({e}); keeping raw mp4v clip (will not play in the "
                      f"dashboard's browser player): {out_path}\n{detail}")
                os.replace(raw_path, out_path)
        print(f"[{self.cfg.CAMERA_ID}] Clip saved: {out_path}")
        # Already running off the hot path, since this method itself is
        # dispatched onto a background thread by its caller, so no need
        # to nest another dispatch just for this notification.
        try:
            self.notifier.send_alert(
                self.cfg.CAMERA_ID, self.cfg.ROOM_NAME,
                "Clip Ready", 0.0, out_path
            )
        except Exception as e:
            print(f"[{self.cfg.CAMERA_ID}] Clip-ready alert email failed: {e}")
        try:
            # Same fix as _dispatch_alert's POST: requests defaults to
            # verifying against the public CA bundle, which a self-signed
            # cert can never pass, so this was silently failing every
            # time once the dashboard served HTTPS -- and being caught by
            # a bare `except: pass` with nothing printed made it doubly
            # invisible. See _dispatch_alert's comment for the full
            # explanation.
            requests.post(
                f"{self.cfg.DASHBOARD_URL}/api/events/add",
                json={
                    "camera_id":  self.cfg.CAMERA_ID,
                    "room":       self.cfg.ROOM_NAME,
                    "event_type": "Clip Ready",
                    "confidence": 0.0,
                    "clip_path":  out_path
                },
                headers={"X-Internal-Key": getattr(self.cfg, "INTERNAL_API_KEY", "")},
                timeout=1.5,
                verify=getattr(self.cfg, "DASHBOARD_CERT_PATH", None) or True,
            )
        except Exception as e:
            print(f"[{self.cfg.CAMERA_ID}] Clip-ready incident history POST failed: {e}")


# class Pipeline used to live here, a thin wrapper that did nothing but
# CameraWorker(config).run(). It has been removed: it had zero test
# coverage, zero callers anywhere in src/ or tests/, and made it too easy
# to mistake CameraWorker.run() for a supported production alternative to
# MultiCameraEngine (it is not; see run()'s docstring above). To visually
# debug one camera without the dashboard, use
# scripts/debug_single_camera.py, which calls CameraWorker(cfg).run()
# directly and is labeled as a dev-only tool.
