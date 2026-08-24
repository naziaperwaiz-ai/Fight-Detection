# src/detection/multi_camera.py
#
# Batched multi-camera inference engine.
#
# Previously, one CameraWorker ran per camera (see pipeline.py) and each
# owned a full copy of the violence classifier and the person/hazard
# detector. N cameras meant N models loaded in memory and N forward
# passes per cycle. Fixed per-call overhead (kernel launch, memory
# transfer) was paid N times, and N threads competed for the same
# CPU/GPU.
#
# MultiCameraEngine owns one copy of each model (violence classifier,
# person/hazard detector, hazard pose model) and, once per cycle, batches
# whichever cameras have a fresh frame into a single forward pass per
# model. Per-camera state (StateMachine, alert/recording bookkeeping,
# track identity, hazard streak counters, motion baseline) stays
# per-camera; only the model calls are shared.
#
# Track identity and hazard streaks are not batched. Ultralytics' built-in
# tracker (model.track(persist=True), used by detector.py's detect()) is
# designed for sequential frames from one continuous stream, and its
# behavior on a fresh batch of independent camera frames each cycle is
# undocumented. A track_id or hazard debounce streak bleeding across
# cameras would corrupt that person's or that room's alerting history.
# Detection, classification, and pose are batched because they are
# stateless per call and batching them changes only performance, not
# correctness. Identity and streak bookkeeping stay small, explicit, and
# per-camera. See detector.py's detect_batch, hazard.py's
# predict_wrists_batch/_fire_events, and pipeline.py's SimpleIOUTracker.
#
# Shared-model limitation: the object classes hazard detection looks for
# (HAZARD_MIN_SEVERITY) and the pose weights/image size used are baked
# into the one shared model and call, so they are the same for every
# hazard-enabled camera in a deployment; they cannot vary per camera the
# way they could when each camera owned its own HazardDetector.
# Per-camera proximity distance and debounce length
# (HAZARD_PROXIMITY_FRAC, HAZARD_MIN_CONSECUTIVE) are post-processing on
# the model's output and do still work per camera.
#
# Capture (reading frames off each camera source) runs on lightweight
# per-camera threads: pure I/O, no model calls. Inference runs on one
# shared loop.

import threading
import time

import cv2
import numpy as np
import torch

from detection.detector import PersonDetector
from detection.pipeline import ViolenceClassifier, CameraWorker, _build_violence_transform


class _CaptureThread(threading.Thread):
    """Reads frames from one camera source into a shared latest-frame
    slot. No model calls happen here, so one thread per camera stays
    cheap regardless of camera count."""

    def __init__(self, cam_id, source):
        super().__init__(daemon=True, name=f"capture-{cam_id}")
        self.cam_id  = cam_id
        self.source  = source
        self.opened  = False
        self._frame  = None
        self._is_new = False
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

    # Consecutive failed reads tolerated before giving up on the current
    # capture handle and reopening the device outright. At the default
    # ~0.1s retry backoff below, 30 is ~3s of sustained failure -- long
    # enough to ride out a single dropped frame, short enough not to
    # sit on a genuinely dead handle for long.
    _MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN = 30

    def run(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        self.opened = cap.isOpened()
        if not self.opened:
            print(f"[{self.cam_id}] Cannot open camera: {self.source}")
            return

        # A single dropped frame -- a transient hiccup with the Windows
        # MSMF backend, a USB glitch, another app briefly grabbing the
        # device -- used to kill this thread outright: cap.read()
        # returning ret=False broke the loop and released the camera for
        # good, with nothing left to ever set self._is_new again. From
        # that point on, MultiCameraEngine._run_cycle's "if not frames:
        # return" silently did nothing every single cycle after that --
        # no detection, no hazard check, no alerts, no live-feed
        # refresh -- with no error printed after the very first warning,
        # so the app looked frozen/"stuck buffering" rather than visibly
        # broken. Retrying a failed read, and reopening the capture
        # outright after enough consecutive failures, keeps one bad
        # frame from taking the whole camera down for the rest of the
        # process's life.
        consecutive_failures = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print(f"[{self.cam_id}] Camera read failed, retrying...")
                if consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
                    print(f"[{self.cam_id}] Camera unresponsive after "
                          f"{consecutive_failures} failed reads, reopening device...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(src)
                    if not cap.isOpened():
                        # Still not back (device unplugged, another app
                        # holding it) -- wait and try the full reopen
                        # again rather than spinning tightly on a handle
                        # that refuses to open.
                        time.sleep(1.0)
                        continue
                    print(f"[{self.cam_id}] Camera reopened.")
                    consecutive_failures = 0
                else:
                    time.sleep(0.1)
                continue
            if consecutive_failures:
                print(f"[{self.cam_id}] Camera recovered after {consecutive_failures} failed read(s).")
            consecutive_failures = 0
            with self._lock:
                self._frame  = frame
                self._is_new = True
        cap.release()

    def get_latest(self):
        """Returns (frame, is_new). is_new is cleared on read. A camera
        that hasn't produced a fresh frame since the last cycle
        contributes nothing this round; its last frame is not
        reprocessed or double-counted as a new sample for motion scoring
        or clip buffering."""
        with self._lock:
            frame, is_new = self._frame, self._is_new
            self._is_new = False
            return frame, is_new

    def stop(self):
        self._stop.set()


class MultiCameraEngine:
    """
    Owns one shared ViolenceClassifier, one shared PersonDetector, and
    (if any camera enables it) one shared HazardDetector. Drives every
    camera through one batched inference loop instead of one
    CameraWorker.run() thread per camera each owning its own model.

    Usage (see main.py):
        engine = MultiCameraEngine(camera_configs)
        engine.start()
        # engine.workers[cam_id] is a CameraWorker. Register it with the
        # dashboard as before; get_frame()/score still work, since
        # process_frame() still sets them.
    """

    def __init__(self, configs, cycle_interval=None):
        if not configs:
            raise ValueError("MultiCameraEngine needs at least one camera config")

        first_cfg   = configs[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model path is assumed the same across all cameras in one
        # deployment: one trained violence classifier, shared. A
        # deployment needing different classifiers per camera should run
        # a second engine instance instead.
        self.model = ViolenceClassifier()
        try:
            state_dict = torch.load(first_cfg.MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            raise RuntimeError(
                "This torch version does not support weights_only=True. "
                "Upgrade torch (>=1.13) before loading model weights. "
                "weights_only=False is a code-execution risk."
            )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"[ENGINE] Shared violence classifier loaded on {self.device} "
              f"for {len(configs)} camera(s)")

        self.detector = PersonDetector(device="cpu")

        self.hazard_detector   = None
        self._hazard_class_map = None
        hazard_cfgs = [c for c in configs if getattr(c, "HAZARD_DETECTION_ENABLED", False)]
        if hazard_cfgs:
            from detection.hazard import HazardDetector, hazard_class_map
            # min_severity/pose weights/imgsz are shared across all
            # hazard-enabled cameras (see module docstring), taken from
            # the first camera that enables hazard detection. Per-camera
            # proximity/debounce settings are not limited this way (see
            # _run_cycle).
            hazard_cfg   = hazard_cfgs[0]
            min_severity = getattr(hazard_cfg, "HAZARD_MIN_SEVERITY", "high")
            self._hazard_class_map = hazard_class_map(min_severity)
            self.hazard_detector = HazardDetector(
                pose_weights=getattr(hazard_cfg, "HAZARD_POSE_WEIGHTS", "yolov8n-pose.pt"),
                device="cpu",
                proximity_frac=getattr(hazard_cfg, "HAZARD_PROXIMITY_FRAC", 0.06),
                min_consecutive=getattr(hazard_cfg, "HAZARD_MIN_CONSECUTIVE", 2),
                imgsz=getattr(hazard_cfg, "HAZARD_IMGSZ", 320),
            )
            print(f"[ENGINE] Shared hazard pose model loaded "
                  f"(classes={self._hazard_class_map}, "
                  f"{len(hazard_cfgs)}/{len(configs)} camera(s) opted in)")

        shared = {
            "model":            self.model,
            "detector":         self.detector,
            "hazard_detector":  self.hazard_detector,
            "hazard_class_map": self._hazard_class_map,
            # One shared preprocessing transform for every worker, instead
            # of each CameraWorker building its own identical T.Compose.
            # See pipeline.py's _build_violence_transform.
            "transform":        _build_violence_transform(),
        }

        # Each camera still gets its own CameraWorker: the state
        # container (StateMachine, alert/recording bookkeeping, motion
        # baseline, its own SimpleIOUTracker) and the object the
        # dashboard's get_frame()/score read from. In shared mode it does
        # not own a model; see pipeline.py's CameraWorker.__init__.
        self.workers              = {}
        self._captures            = {}
        self._hazard_frame_counts = {}   # cam_id -> int, per-camera HAZARD_SAMPLE_EVERY_N_FRAMES throttle
        self._hazard_streaks      = {}   # cam_id -> {"_any_hazard_object": streak}, see hazard.py's _fire_events
        # Rate-limits the "hazard candidate seen" / "candidate seen but no
        # event fired" diagnostics below, so a knife sitting in frame for
        # a while logs periodically, not once per due sample (~3/sec).
        # See _run_cycle steps 3 and 5 for why these exist: previously, a
        # hazard object being seen by the detector but never turning into
        # a fired event (wrist not detected by the pose model, or
        # detected but not close enough) produced zero console output,
        # making "the detector never saw it" and "it saw it but the
        # proximity/debounce check failed" indistinguishable from the
        # outside.
        self._last_hazard_seen_log  = {}   # cam_id -> time.time() of last "candidate seen" log
        self._last_hazard_miss_log  = {}   # cam_id -> time.time() of last "seen but no event fired" log
        # Rate-limits the "cycle is running slower than its target
        # interval" diagnostic in _run_cycle below. Exists because a
        # cycle that takes seconds instead of a fraction of a second
        # means process_frame() is being called far less often than
        # config.FPS implies for every camera in this engine -- which
        # starves buffer_times of real samples and, downstream, can
        # leave a saved clip covering far fewer real frames than its
        # recording window's wall-clock length suggests (see pipeline.py's
        # _save_clip). Logging which step (detection / classifier /
        # hazard pose) is actually slow turns "the clip looks wrong" into
        # a concrete number instead of a guess.
        self._last_slow_cycle_log = 0.0
        for cfg in configs:
            self.workers[cfg.CAMERA_ID]              = CameraWorker(cfg, shared=shared)
            self._captures[cfg.CAMERA_ID]             = _CaptureThread(cfg.CAMERA_ID, cfg.CAMERA_SOURCE)
            self._hazard_frame_counts[cfg.CAMERA_ID]  = 0
            self._hazard_streaks[cfg.CAMERA_ID]       = {}
            self._last_hazard_seen_log[cfg.CAMERA_ID] = 0.0
            self._last_hazard_miss_log[cfg.CAMERA_ID] = 0.0

        self._stop_event = threading.Event()
        self._thread      = None
        # How often the inference loop gathers a batch and runs.
        # Defaults to the fastest configured camera's frame interval so
        # no stream is bottlenecked below its own FPS setting.
        self.cycle_interval = cycle_interval or (1.0 / max(c.FPS for c in configs))

    def start(self):
        for cap in self._captures.values():
            cap.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="engine-inference")
        self._thread.start()
        print(f"[ENGINE] Running, cycle_interval={self.cycle_interval:.3f}s "
              f"({len(self.workers)} camera(s))")
        return self

    def stop(self):
        self._stop_event.set()
        for cap in self._captures.values():
            cap.stop()

    def _run_loop(self):
        while not self._stop_event.is_set():
            cycle_start = time.time()
            try:
                self._run_cycle()
            except Exception as e:
                # One bad cycle (a malformed frame, a transient model
                # error) should not take down every camera's inference.
                # Log it and keep going, matching the network-call
                # try/excepts elsewhere in this pipeline.
                print(f"[ENGINE] Cycle failed, continuing: {e}")
            elapsed = time.time() - cycle_start
            if elapsed < self.cycle_interval:
                time.sleep(self.cycle_interval - elapsed)

    def _run_cycle(self):
        _t0 = time.time()   # see the slow-cycle diagnostic at the end of this method
        # 1. Gather whichever cameras have a genuinely new frame this
        # cycle. A camera slower than the cycle rate contributes nothing
        # this round; its stale frame is not reprocessed.
        cam_ids, frames = [], []
        for cam_id, cap in self._captures.items():
            frame, is_new = cap.get_latest()
            if frame is not None and is_new:
                cam_ids.append(cam_id)
                frames.append(frame)
        if not frames:
            return
        frame_by_cam = dict(zip(cam_ids, frames))

        for cam_id, frame in frame_by_cam.items():
            self.workers[cam_id].buffer.append(frame.copy())
            # Kept in lockstep with buffer itself -- see CameraWorker's
            # buffer_times docstring in pipeline.py -- so a saved clip's
            # writer FPS can reflect this camera's real achieved capture
            # rate under this engine's batched cycle, not just assume
            # config.FPS was hit.
            self.workers[cam_id].buffer_times.append(time.time())

        _t1 = time.time()
        # 2. One batched person (and hazard-object) detection call
        # across every camera with a fresh frame this cycle, replacing
        # one detector call per camera.
        want_hazard = self.hazard_detector is not None
        batch_results = self.detector.detect_batch(
            frames, extra_classes=self._hazard_class_map if want_hazard else None
        )
        _t2 = time.time()

        # 3. Per camera: assign track ids with that camera's own
        # SimpleIOUTracker (cheap, no model, correct to keep per-camera;
        # see its docstring), gate on that camera's own motion score, and
        # collect crops into one shared list for a single classifier
        # batch across all cameras.
        per_cam_detections = {}   # cam_id -> [(track_id, bbox, crop), ...]
        per_cam_motion_ok  = {}
        all_crops          = []
        crop_owner         = []   # parallel to all_crops: (cam_id, track_id)
        hazard_candidates_by_cam = {}

        for cam_id, result in zip(cam_ids, batch_results):
            worker = self.workers[cam_id]
            frame  = frame_by_cam[cam_id]
            raw_dets, extra_dets = result if want_hazard else (result, [])

            boxes     = [bbox for bbox, crop in raw_dets]
            track_ids = worker.tracker.update(boxes)
            dets = [(tid, bbox, crop) for tid, (bbox, crop) in zip(track_ids, raw_dets)]
            per_cam_detections[cam_id] = dets

            motion_ok = worker._motion_score(frame) >= worker.cfg.MOTION_THRESHOLD
            per_cam_motion_ok[cam_id] = motion_ok
            if motion_ok:
                for tid, bbox, crop in dets:
                    all_crops.append(crop)
                    crop_owner.append((cam_id, tid))

            # want_hazard is engine-wide (true if any camera opted in)
            # and only controls whether extra_classes is passed to the
            # shared detect_batch call. Whether this camera's own
            # extra_dets are used downstream still depends on this
            # camera's own HAZARD_DETECTION_ENABLED, so a camera that
            # did not opt in cannot have hazard events fire for it
            # because a different camera in the same engine did.
            if want_hazard and getattr(worker.cfg, "HAZARD_DETECTION_ENABLED", False):
                hazard_candidates_by_cam[cam_id] = extra_dets
                if extra_dets:
                    now = time.time()
                    if now - self._last_hazard_seen_log[cam_id] > 2.0:
                        self._last_hazard_seen_log[cam_id] = now
                        labels = ", ".join(f"{lbl}({conf:.2f})" for lbl, conf, _ in extra_dets)
                        print(f"[{cam_id}] Hazard-class object(s) detected this frame: {labels}")

        _t3 = time.time()
        # 4. One batched violence-classifier forward pass across every
        # camera's crops, replacing a call per camera per person. It runs
        # once per cycle, covering every tracked person on every camera.
        scores_by_owner = {}
        if all_crops:
            tensors = [
                self.workers[cam_id].transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                for (cam_id, _tid), crop in zip(crop_owner, all_crops)
            ]
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                out   = self.model(batch_tensor)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            for (cam_id, tid), p in zip(crop_owner, probs):
                scores_by_owner[(cam_id, tid)] = float(p)
        _t4 = time.time()

        # 5. Hazard pose, batched across only the cameras that are due a
        # sample per their own HAZARD_SAMPLE_EVERY_N_FRAMES throttle.
        # Same gating as the single-camera path, evaluated per camera
        # before the batch is built instead of after: due is on the
        # sample throttle alone, not on whether a candidate object is
        # present, so a due camera with no candidate object still gets
        # its streak cleared below instead of leaving a stale count to
        # resume later (see hazard.py's _fire_events for the same
        # concern on the single-camera path via check_objects()).
        hazard_events_by_cam = {cam_id: [] for cam_id in cam_ids}
        if want_hazard:
            due_cam_ids, due_frames, due_prox_px, due_min_consec, due_objects = [], [], [], [], []
            for cam_id in cam_ids:
                extra_dets = hazard_candidates_by_cam.get(cam_id, [])
                self._hazard_frame_counts[cam_id] += 1
                sample_every = max(1, getattr(self.workers[cam_id].cfg, "HAZARD_SAMPLE_EVERY_N_FRAMES", 5))
                if self._hazard_frame_counts[cam_id] % sample_every != 0:
                    continue
                if not extra_dets:
                    # Due sample, no candidate hazard object this cycle:
                    # clear this camera's streak now rather than letting
                    # it survive the gap and resume counting on the next
                    # detection.
                    self._hazard_streaks[cam_id].clear()
                    continue
                frame = frame_by_cam[cam_id]
                h, w  = frame.shape[:2]
                # Proximity fraction and debounce length are respected
                # per camera even though the pose model itself is shared;
                # see module docstring.
                prox_frac  = getattr(self.workers[cam_id].cfg, "HAZARD_PROXIMITY_FRAC", 0.06)
                min_consec = getattr(self.workers[cam_id].cfg, "HAZARD_MIN_CONSECUTIVE", 2)
                due_cam_ids.append(cam_id)
                due_frames.append(frame)
                due_prox_px.append(prox_frac * float(np.hypot(h, w)))
                due_min_consec.append(min_consec)
                due_objects.append(extra_dets)

            if due_frames:
                try:
                    wrists_batch = self.hazard_detector.predict_wrists_batch(due_frames)
                except Exception as e:
                    print(f"[ENGINE] Hazard pose batch failed, skipping this cycle's sample: {e}")
                    wrists_batch = [[] for _ in due_frames]

                from detection.hazard import _fire_events
                for cam_id, objs, wrists, prox_px, min_consec in zip(
                    due_cam_ids, due_objects, wrists_batch, due_prox_px, due_min_consec
                ):
                    events = _fire_events(
                        objs, wrists, prox_px, min_consec, self._hazard_streaks[cam_id]
                    )
                    hazard_events_by_cam[cam_id] = events
                    # Logging, notifier.send_alert, and the
                    # /api/events/add POST for any fired event happen
                    # inside process_frame() below (step 6), not here, so
                    # there is exactly one place that reacts to a fired
                    # hazard event, matching the single-camera path.

                    # Diagnostic: a candidate object was seen on this due
                    # sample but no event fired -- either the pose model
                    # found no wrist at all this sample (0 wrist points),
                    # or it found one but not within HAZARD_PROXIMITY_FRAC
                    # of the object, or the streak just hasn't reached
                    # HAZARD_MIN_CONSECUTIVE yet. Rate-limited per camera
                    # since a knife held in frame stays "seen" across many
                    # due samples in a row.
                    if objs and not events:
                        now = time.time()
                        if now - self._last_hazard_miss_log[cam_id] > 2.0:
                            self._last_hazard_miss_log[cam_id] = now
                            print(f"[{cam_id}] Hazard candidate present but no event fired "
                                  f"this sample: {len(wrists)} wrist point(s) detected, "
                                  f"streak={dict(self._hazard_streaks[cam_id])}, "
                                  f"proximity_px={prox_px:.0f}, min_consecutive={min_consec}")

        _t5 = time.time()
        # 6. Dispatch: per-camera alerting, recording, and drawing via
        # CameraWorker.process_frame(), the single implementation of
        # "what an alert means" shared by both paths.
        for cam_id in cam_ids:
            worker    = self.workers[cam_id]
            frame     = frame_by_cam[cam_id]
            dets      = per_cam_detections[cam_id]
            motion_ok = per_cam_motion_ok[cam_id]
            scores_by_tid = {
                tid: scores_by_owner.get((cam_id, tid), 0.0)
                for tid, bbox, crop in dets
            }
            worker.process_frame(frame, dets, motion_ok, scores_by_tid, hazard_events_by_cam.get(cam_id, []))

        # Diagnostic: a cycle that takes noticeably longer than its
        # target cycle_interval means process_frame() is running for
        # every camera far less often than config.FPS implies. That
        # directly starves buffer_times of real samples -- see
        # pipeline.py's _save_clip -- so a saved clip can end up
        # covering far fewer real frames than its recording window's
        # wall-clock length suggests, even after fixing the writer to
        # use the real achieved fps instead of assuming config.FPS.
        # Logged at most once every 5s (not every cycle) so a
        # consistently slow deployment doesn't flood the console, and
        # only when a cycle runs at least 3x its target -- a cycle
        # finishing a little late under normal jitter is not worth
        # logging. Breaks the total down by step so "the pose model is
        # slow" and "the person detector is slow" are distinguishable
        # from the console alone, rather than needing a profiler.
        _t6 = time.time()
        total = _t6 - _t0
        if total > max(1.0, self.cycle_interval * 3) and (_t6 - self._last_slow_cycle_log) > 5.0:
            self._last_slow_cycle_log = _t6
            print(f"[ENGINE] Cycle took {total:.2f}s (target {self.cycle_interval:.3f}s) for "
                  f"{len(cam_ids)} camera(s) -- gather={_t1-_t0:.2f}s detect={_t2-_t1:.2f}s "
                  f"track/motion={_t3-_t2:.2f}s classify={_t4-_t3:.2f}s hazard_pose={_t5-_t4:.2f}s "
                  f"dispatch={_t6-_t5:.2f}s")
