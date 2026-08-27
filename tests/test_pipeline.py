# tests/test_pipeline.py
#
# Tests for detection/pipeline.py's process_frame() alert-escalation
# wiring that is not already covered by test_multi_camera.py:
#   - has_fighting() previously had zero callers anywhere in the
#     codebase; this confirms process_frame() now dispatches a
#     "Fighting Detected" alert when the state machine reports the
#     Fighting state, and that it shares the same last_alert_time
#     cooldown budget as every other alert type on the camera (not a
#     separate one), matching has_emergency()/has_fall()'s existing
#     behavior.
#
#   PYTHONPATH=src pytest tests/test_pipeline.py -v

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from detection import pipeline as pipeline_mod  # noqa: E402
from detection.pipeline import CameraWorker, ViolenceClassifier  # noqa: E402
from detection.detector import PersonDetector  # noqa: E402
from detection.config import Config  # noqa: E402


def _make_worker(tmp_path):
    cfg = Config()
    cfg.MODEL_PATH        = None   # unused; shared mode does not load a checkpoint
    cfg.CAMERA_ID         = "CAM-X"
    cfg.ROOM_NAME         = "Room X"
    cfg.CAMERA_SOURCE     = 0
    cfg.MOTION_THRESHOLD  = 999999.0
    cfg.FPS               = 15
    cfg.BUFFER_SECONDS    = 2
    cfg.CONFIRM_SECONDS   = 1
    cfg.DASHBOARD_URL     = "http://127.0.0.1:1"
    cfg.EMAIL_SENDER      = "test@example.com"
    cfg.EMAIL_APP_PASSWORD = "x"
    cfg.EMAIL_RECIPIENTS  = []
    cfg.COOLDOWN_SECONDS  = 120
    cfg.VIOLENCE_THRESHOLD = 0.99
    cfg.POST_EVENT_SECONDS = 5
    cfg.CLIPS_DIR         = str(tmp_path)

    shared = {
        "model": ViolenceClassifier(),
        "detector": PersonDetector.__new__(PersonDetector),   # not exercised by process_frame directly
        "hazard_detector": None,
        "hazard_class_map": None,
    }
    worker = CameraWorker(cfg, shared=shared)
    # No real settings file exists at the path notifier.py computes in a
    # test run; load_alert_settings() already degrades to {} in that
    # case, so cooldown falls back to cfg.COOLDOWN_SECONDS as asserted
    # below. Nothing to stub for that.
    return worker


def test_fighting_state_dispatches_an_alert(tmp_path, monkeypatch):
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: True)

    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda notifier_args, api_payload: dispatched.append((notifier_args, api_payload)))

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    worker.process_frame(frame, detections=[], motion_ok=False)

    fighting_calls = [c for c in dispatched if c[1]["event_type"] == "Fighting Detected"]
    assert len(fighting_calls) == 1
    notifier_args, api_payload = fighting_calls[0]
    assert notifier_args[2] == "Fighting Detected"
    assert api_payload["camera_id"] == "CAM-X"
    assert api_payload["room"] == "Room X"


def test_fighting_alert_respects_the_shared_cooldown(tmp_path, monkeypatch):
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: True)

    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda notifier_args, api_payload: dispatched.append(api_payload))

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    worker.process_frame(frame, detections=[], motion_ok=False)
    assert len([c for c in dispatched if c["event_type"] == "Fighting Detected"]) == 1

    # Immediately fires again while still in the Fighting state: the
    # cooldown set by the first alert must suppress a second one right
    # away, the same as it would for Emergency or Fall.
    worker.process_frame(frame, detections=[], motion_ok=False)
    assert len([c for c in dispatched if c["event_type"] == "Fighting Detected"]) == 1


def test_fighting_alert_shares_cooldown_budget_with_emergency(tmp_path, monkeypatch):
    # Regression guard for the "one alerting budget per camera" design:
    # an Emergency alert firing first must still block an
    # immediately-following Fighting alert, since both read/write the
    # same self.last_alert_time.
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_emergency", lambda: True)
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: True)

    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda notifier_args, api_payload: dispatched.append(api_payload))

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    worker.process_frame(frame, detections=[], motion_ok=False)

    event_types = [c["event_type"] for c in dispatched]
    assert event_types.count("Emergency") == 1
    assert event_types.count("Fighting Detected") == 0


# ---------------------------------------------------------------------
# Live-feed privacy blur: get_frame()/video_feed must stay a frozen,
# pixelated placeholder except while something is actually happening
# (Agitated and up, a confirmed fall, or a hazard event this frame), and
# must re-blur LIVE_BLUR_HYSTERESIS_SECONDS after the last such trigger.
# process_frame()'s own return value (used by the debug tool's
# cv2.imshow, not the dashboard) is untouched by any of this; only what
# gets published via _set_frame/get_frame is affected.
# ---------------------------------------------------------------------

def _varied_frame():
    # A patterned, non-uniform frame so pixelation is actually detectable
    # (a blank/uniform frame would look "blurred" even unblurred).
    return np.tile(np.arange(64, dtype=np.uint8).reshape(1, 64, 1), (64, 1, 3))


def _fake_clock(monkeypatch, start=1_000_000.0):
    clock = {"t": start}
    monkeypatch.setattr(pipeline_mod.time, "time", lambda: clock["t"])
    return clock


def test_live_feed_defaults_to_blurred_before_any_trigger(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    frame = _varied_frame()

    worker.process_frame(frame, detections=[], motion_ok=False)

    published = worker.get_frame()
    assert not np.array_equal(published, frame)
    # Pixelated to a 2x2 block grid on a 64x64 frame: far fewer distinct
    # rows of color than the original 64-value gradient.
    assert len(np.unique(published)) < len(np.unique(frame))


def test_live_feed_unblurs_when_a_track_is_agitated(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker.state_machine._get_or_create(1).state = "Agitated"
    frame = _varied_frame()

    worker.process_frame(frame, detections=[], motion_ok=False)

    assert np.array_equal(worker.get_frame(), frame)


def test_live_feed_stays_blurred_on_proximate_alone(tmp_path, monkeypatch):
    # Proximate (people simply near each other) is deliberately not a
    # trigger; only Agitated and above should unblur.
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker.state_machine._get_or_create(1).state = "Proximate"
    frame = _varied_frame()

    worker.process_frame(frame, detections=[], motion_ok=False)

    assert not np.array_equal(worker.get_frame(), frame)


def test_live_feed_unblurs_on_hazard_event_this_frame(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    frame = _varied_frame()

    worker.process_frame(
        frame, detections=[], motion_ok=False,
        hazard_events=[{"object": "knife", "detection_conf": 0.9, "severity": "high", "detail": "test"}],
    )

    assert np.array_equal(worker.get_frame(), frame)


def test_live_feed_unblurs_on_confirmed_fall(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fall", lambda: True)
    frame = _varied_frame()

    worker.process_frame(frame, detections=[], motion_ok=False)

    assert np.array_equal(worker.get_frame(), frame)


def test_live_feed_reblurs_after_hysteresis_window_elapses(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker._live_blur_hysteresis = 15
    frame = _varied_frame()

    worker.state_machine._get_or_create(1).state = "Agitated"
    worker.process_frame(frame, detections=[], motion_ok=False)
    assert np.array_equal(worker.get_frame(), frame)   # unblurred at the moment of the trigger

    # Trigger clears (back to Normal), and less than the hysteresis
    # window has passed: still unblurred.
    worker.state_machine.tracks[1].state = "Normal"
    clock["t"] += 10
    worker.process_frame(frame, detections=[], motion_ok=False)
    assert np.array_equal(worker.get_frame(), frame)

    # Past the hysteresis window since the last trigger: back to blurred.
    clock["t"] += 10   # 20s since the trigger, past the 15s window
    worker.process_frame(frame, detections=[], motion_ok=False)
    assert not np.array_equal(worker.get_frame(), frame)


def test_live_feed_blurred_placeholder_is_frozen_until_refresh_interval(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker._live_blur_refresh = 30

    frame_1 = _varied_frame()
    worker.process_frame(frame_1, detections=[], motion_ok=False)
    published_1 = worker.get_frame()

    # A different real frame arrives, but within the refresh interval:
    # the published placeholder must not change (frozen).
    clock["t"] += 5
    frame_2 = np.full((64, 64, 3), 250, dtype=np.uint8)
    worker.process_frame(frame_2, detections=[], motion_ok=False)
    assert np.array_equal(worker.get_frame(), published_1)

    # Past the refresh interval: the placeholder is regenerated from
    # whatever the current real frame is.
    clock["t"] += 30
    worker.process_frame(frame_2, detections=[], motion_ok=False)
    published_3 = worker.get_frame()
    assert not np.array_equal(published_3, published_1)


def test_debug_run_loop_return_value_is_never_blurred(tmp_path, monkeypatch):
    # process_frame()'s return value feeds the dev-only run() loop's
    # cv2.imshow window directly, not get_frame()/video_feed. It must
    # always be the real annotated frame, blur state notwithstanding.
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    frame = _varied_frame()

    returned = worker.process_frame(frame, detections=[], motion_ok=False)
    assert np.array_equal(returned, frame)
    assert not np.array_equal(worker.get_frame(), frame)   # dashboard copy is still blurred


# ---------------------------------------------------------------------
# Hazard box visualization: process_frame() must draw a visible box on
# the actual flagged object (ev["bbox"]) rather than only logging/
# alerting on a hazard event, so a caregiver looking at the live feed
# sees what was flagged, not just a bounding box on the tracked person.
# Uses process_frame()'s returned frame directly (not get_frame(),
# which may still be showing the frozen blur placeholder) since
# _draw_hazard_boxes() runs before _publish_live_frame() either way.
# ---------------------------------------------------------------------

def _hazard_event(bbox=(10.0, 10.0, 30.0, 30.0), obj="knife"):
    return {
        "object": obj, "severity": "high", "detection_conf": 0.9,
        "wrist_distance_px": 12.0, "detail": "test", "bbox": bbox,
    }


def test_hazard_event_draws_a_box_on_the_flagged_object(tmp_path, monkeypatch):
    # cv2's draw calls inside process_frame() mutate the frame array
    # in place, so the caller's `frame` variable ends up equal to the
    # return value regardless of what was drawn -- comparing against a
    # pristine copy taken before the call is what actually proves
    # something was drawn.
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    frame = _varied_frame()
    pristine = frame.copy()

    returned = worker.process_frame(
        frame, detections=[], motion_ok=False,
        hazard_events=[_hazard_event()],
    )

    assert not np.array_equal(returned, pristine)
    assert len(worker._flagged_hazard_boxes) == 1
    bbox, label, expires_at = worker._flagged_hazard_boxes[0]
    assert bbox == (10.0, 10.0, 30.0, 30.0)
    assert label == "knife"


def test_hazard_event_without_bbox_is_skipped_not_raised(tmp_path, monkeypatch):
    # Older event dicts (e.g. hand-built in other tests) may not carry
    # "bbox" at all; drawing a box is a visual nicety and must not be
    # able to break process_frame() when it's missing.
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    frame = _varied_frame()

    worker.process_frame(
        frame, detections=[], motion_ok=False,
        hazard_events=[{"object": "knife", "detection_conf": 0.9, "severity": "high", "detail": "test"}],
    )

    assert worker._flagged_hazard_boxes == []


def test_hazard_box_persists_within_display_window(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker._hazard_box_display_seconds = 5

    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[_hazard_event()])
    assert len(worker._flagged_hazard_boxes) == 1

    # Still within the display window, no new hazard event this frame:
    # the box from the earlier event must still be held (not pruned).
    clock["t"] += 3
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[])
    assert len(worker._flagged_hazard_boxes) == 1


def test_hazard_box_expires_after_display_window(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker._hazard_box_display_seconds = 5

    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[_hazard_event()])
    assert len(worker._flagged_hazard_boxes) == 1

    # Past the display window, no new hazard event: the stale box must
    # be pruned, not held forever.
    clock["t"] += 10
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[])
    assert worker._flagged_hazard_boxes == []


def test_multiple_hazard_events_each_get_their_own_box(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    frame = _varied_frame()

    worker.process_frame(
        frame, detections=[], motion_ok=False,
        hazard_events=[
            _hazard_event(bbox=(1.0, 1.0, 5.0, 5.0), obj="knife"),
            _hazard_event(bbox=(40.0, 40.0, 50.0, 50.0), obj="scissors"),
        ],
    )

    labels = sorted(label for _, label, _ in worker._flagged_hazard_boxes)
    assert labels == ["knife", "scissors"]


# ---------------------------------------------------------------------
# Hazard events must produce a saved clip, the same as a violence alert
# does, not just an email/incident log entry with no visual record.
# Recording is started outside the cooldown gate (same reasoning as the
# hazard box flag above) but guarded on alert_active, so a recording
# already in progress -- from this hazard event or from a violence
# alert -- is never interrupted or its buffer reset by another event
# firing mid-recording.
# ---------------------------------------------------------------------

def test_hazard_event_starts_recording(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker.buffer.append(np.zeros((4, 4, 3), dtype=np.uint8))
    worker.buffer.append(np.ones((4, 4, 3), dtype=np.uint8))

    assert worker.recording is False
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[_hazard_event()])

    assert worker.recording is True
    assert worker.alert_active is True
    assert worker.record_start_time == clock["t"]
    # The pre-event buffer (2 frames) seeds the clip, same as the
    # violence-alert path's list(self.buffer); process_frame then
    # appends this frame itself before returning, landing at 3.
    assert len(worker.record_frames) == 3


def test_hazard_event_does_not_interrupt_an_in_progress_recording(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)

    # Simulate a violence alert already mid-recording.
    worker.alert_active      = True
    worker.recording         = True
    worker.record_start_time = clock["t"]
    sentinel_frames = [np.full((4, 4, 3), 42, dtype=np.uint8)]
    worker.record_frames = sentinel_frames

    clock["t"] += 2
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[_hazard_event()])

    # record_start_time and record_frames (the sentinel list object
    # itself, not just its contents) must be untouched -- _start_recording
    # must not have been called a second time.
    assert worker.record_start_time == clock["t"] - 2
    assert worker.record_frames is sentinel_frames


def test_hazard_event_eventually_saves_a_clip(tmp_path, monkeypatch):
    # End-to-end through the same POST_EVENT_SECONDS countdown the
    # violence-alert path uses: once enough time has passed since
    # recording started, a clip must actually get written to disk.
    # threading.Thread is replaced with something that runs its target
    # synchronously, so the save is guaranteed to have happened by the
    # time this test asserts on it, rather than racing a real background
    # thread. _dispatch_alert is stubbed out for the same reason: with
    # threading patched to run synchronously, an un-stubbed
    # _dispatch_alert would perform a real SMTP connection attempt and
    # HTTP POST inline, inside this test, at the mercy of whatever this
    # environment's outbound network happens to do (observed to hang
    # for minutes rather than fail fast). Recording/clip-saving is what
    # this test is about; alert dispatch has its own coverage elsewhere.
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)   # CameraWorker.__init__ itself needs a real threading.Lock
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    worker.cfg.POST_EVENT_SECONDS = 1
    saved = []
    monkeypatch.setattr(worker, "_save_clip", lambda frames, elapsed_seconds=None, confidence=None: saved.append(frames))

    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[_hazard_event()])
    assert worker.recording is True

    clock["t"] += 2   # past POST_EVENT_SECONDS
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[])

    assert worker.recording is False
    assert worker.alert_active is False
    assert len(saved) == 1
    assert len(saved[0]) >= 1   # a real, non-empty frame list was handed off


def test_hazard_clip_reports_the_triggering_detection_confidence(tmp_path, monkeypatch):
    # Regression test: _save_clip's "Clip Ready" incident used to always
    # report a hardcoded 0.0 confidence, regardless of what actually
    # triggered the recording -- so a real 90%-confidence knife
    # detection would show up in Incident History as a 0% entry right
    # next to the "Hazard Detected" incident that fired at 90%,
    # reading as a failed detection when nothing failed. _start_recording
    # now threads the triggering event's confidence through to
    # _save_clip; this asserts that value is what actually gets passed,
    # not just that _save_clip was called at all (see the test above).
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    worker.cfg.POST_EVENT_SECONDS = 1
    captured = []
    monkeypatch.setattr(
        worker, "_save_clip",
        lambda frames, elapsed_seconds=None, confidence=None: captured.append(confidence),
    )

    worker.process_frame(_varied_frame(), detections=[], motion_ok=False,
                          hazard_events=[_hazard_event()])   # detection_conf=0.9
    clock["t"] += 2   # past POST_EVENT_SECONDS
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[])

    assert captured == [0.9]


# ---------------------------------------------------------------------
# Hazard supervision context: a hazard object near a wrist while someone
# else is also in frame (a caregiver, most likely) reads as normal care
# activity, not something worth paging a caregiver about -- see the
# HAZARD_REQUIRE_UNSUPERVISED design note in process_frame. person_count
# comes from `detections`, the same per-frame tracked-people list every
# other part of process_frame already receives, not a new detector call.
# ---------------------------------------------------------------------

def _two_person_detections():
    return [("p1", (0, 0, 10, 10), None), ("p2", (20, 20, 30, 30), None)]


def test_hazard_event_suppressed_when_supervised(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: dispatched.append(a))

    worker.process_frame(
        _varied_frame(), detections=_two_person_detections(), motion_ok=False,
        hazard_events=[_hazard_event()],
    )

    assert dispatched == []
    assert worker.recording is False
    assert worker._flagged_hazard_boxes == []


def test_hazard_event_fires_when_alone(tmp_path, monkeypatch):
    # Regression check that the supervision gate doesn't also suppress
    # the case it exists to let through: exactly one tracked person
    # (the patient, alone) must still fire normally.
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: dispatched.append(a))

    worker.process_frame(
        _varied_frame(), detections=[("p1", (0, 0, 10, 10), None)], motion_ok=False,
        hazard_events=[_hazard_event()],
    )

    assert len(dispatched) == 1
    assert worker.recording is True


def test_hazard_require_unsupervised_false_still_fires_when_supervised(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker.cfg.HAZARD_REQUIRE_UNSUPERVISED = False
    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: dispatched.append(a))

    worker.process_frame(
        _varied_frame(), detections=_two_person_detections(), motion_ok=False,
        hazard_events=[_hazard_event()],
    )

    assert len(dispatched) == 1
    assert worker.recording is True


def test_hazard_quiet_hours_marks_the_dispatched_event(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker.cfg.HAZARD_QUIET_HOURS_START = 22
    worker.cfg.HAZARD_QUIET_HOURS_END   = 6

    class _FakeDatetime:
        @staticmethod
        def now():
            return SimpleNamespace(hour=2)   # 2am -- inside the 22->6 window
    monkeypatch.setattr(pipeline_mod, "datetime", _FakeDatetime)

    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda notifier_args, api_payload: dispatched.append(api_payload))

    worker.process_frame(
        _varied_frame(), detections=[("p1", (0, 0, 10, 10), None)], motion_ok=False,
        hazard_events=[_hazard_event()],
    )

    assert len(dispatched) == 1
    assert dispatched[0]["quiet_hours"] is True
    assert "quiet hours" in dispatched[0]["detail"]


def test_hazard_outside_quiet_hours_is_not_marked(tmp_path, monkeypatch):
    _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    worker.cfg.HAZARD_QUIET_HOURS_START = 22
    worker.cfg.HAZARD_QUIET_HOURS_END   = 6

    class _FakeDatetime:
        @staticmethod
        def now():
            return SimpleNamespace(hour=14)   # 2pm -- outside the 22->6 window
    monkeypatch.setattr(pipeline_mod, "datetime", _FakeDatetime)

    dispatched = []
    monkeypatch.setattr(worker, "_dispatch_alert", lambda notifier_args, api_payload: dispatched.append(api_payload))

    worker.process_frame(
        _varied_frame(), detections=[("p1", (0, 0, 10, 10), None)], motion_ok=False,
        hazard_events=[_hazard_event()],
    )

    assert len(dispatched) == 1
    assert dispatched[0]["quiet_hours"] is False
    assert "quiet hours" not in dispatched[0]["detail"]


# ---------------------------------------------------------------------
# Clip recording for Fighting/Fall/Emergency: these three previously
# never called _start_recording, so their incidents always showed "No
# clip" in Incident History even though Violence Detected and Hazard
# Detected both got a saved clip. Same pattern as the hazard tests
# above: start outside the cooldown gate, guarded only on alert_active,
# and eventually hand off to _save_clip via the POST_EVENT_SECONDS
# countdown.
# ---------------------------------------------------------------------

def test_fighting_event_starts_recording(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: True)
    worker.buffer.append(np.zeros((4, 4, 3), dtype=np.uint8))
    worker.buffer.append(np.ones((4, 4, 3), dtype=np.uint8))

    assert worker.recording is False
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False)

    assert worker.recording is True
    assert worker.alert_active is True
    assert worker.record_start_time == clock["t"]


def test_fall_event_starts_recording(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fall", lambda: True)
    worker.buffer.append(np.zeros((4, 4, 3), dtype=np.uint8))
    worker.buffer.append(np.ones((4, 4, 3), dtype=np.uint8))

    assert worker.recording is False
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False)

    assert worker.recording is True
    assert worker.alert_active is True
    assert worker.record_start_time == clock["t"]


def test_emergency_event_starts_recording(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_emergency", lambda: True)
    worker.buffer.append(np.zeros((4, 4, 3), dtype=np.uint8))
    worker.buffer.append(np.ones((4, 4, 3), dtype=np.uint8))

    assert worker.recording is False
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False)

    assert worker.recording is True
    assert worker.alert_active is True
    assert worker.record_start_time == clock["t"]


def test_fighting_event_does_not_interrupt_an_in_progress_recording(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: True)

    # Simulate a violence alert already mid-recording.
    worker.alert_active      = True
    worker.recording         = True
    worker.record_start_time = clock["t"]
    sentinel_frames = [np.full((4, 4, 3), 42, dtype=np.uint8)]
    worker.record_frames = sentinel_frames

    clock["t"] += 2
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False)

    assert worker.record_start_time == clock["t"] - 2
    assert worker.record_frames is sentinel_frames


def test_fighting_event_eventually_saves_a_clip(tmp_path, monkeypatch):
    # _dispatch_alert is stubbed out here (unlike the hazard version of
    # this test) because threading is patched below to run everything
    # synchronously, including _dispatch_alert's own background send --
    # left un-stubbed, that would make this test perform a real SMTP
    # connection attempt and HTTP POST inline, at the mercy of whatever
    # this environment's outbound network happens to do. Recording/
    # clip-saving is what this test is actually about; alert dispatch is
    # exercised separately by test_fighting_state_dispatches_an_alert.
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: True)
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    worker.cfg.POST_EVENT_SECONDS = 1
    saved = []
    monkeypatch.setattr(worker, "_save_clip", lambda frames, elapsed_seconds=None, confidence=None: saved.append(frames))

    worker.process_frame(_varied_frame(), detections=[], motion_ok=False)
    assert worker.recording is True

    clock["t"] += 2   # past POST_EVENT_SECONDS
    monkeypatch.setattr(worker.state_machine, "has_fighting", lambda: False)
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False)

    assert worker.recording is False
    assert worker.alert_active is False
    assert len(saved) == 1
    assert len(saved[0]) >= 1


# ---------------------------------------------------------------------
# _save_clip's writer FPS: regression coverage for a real production
# bug. The clip writer used to always use config.FPS -- the camera's
# *configured* target rate -- regardless of how many frames were
# actually captured during the real pre-event-buffer-plus-recording
# window. Under MultiCameraEngine's batched cycle, the real per-camera
# capture/append rate can fall well short of config.FPS under load
# (more cameras, hazard pose detection enabled, a slow host). Writing
# far fewer frames than config.FPS implies, at config.FPS, produces a
# technically valid .mp4 whose reported duration (frame_count / fps) is
# a small fraction of how long the window actually spanned -- rounding
# to "0:00" in most players for a short enough frame count, even though
# a real multi-second event was captured. _save_clip now derives the
# writer's fps from elapsed_seconds (real wall-clock span) and the
# actual frame count instead, falling back to config.FPS only when that
# span is missing or too small to trust.
# ---------------------------------------------------------------------

def _capture_writer_fps(monkeypatch, worker):
    # _save_clip does two more things after writing the file: a real
    # email send (notifier.send_alert) and a real HTTP POST
    # (requests.post) to announce "Clip Ready". Both are genuine network
    # I/O -- left un-mocked, a test calling _save_clip directly (not
    # through process_frame's background-thread dispatch) blocks on them
    # inline, at the mercy of this environment's outbound network. Same
    # class of hang already fixed once in the hazard/fighting
    # eventually-saves-a-clip tests above; mocked here too so a test
    # about writer fps doesn't also become a flaky network test.
    monkeypatch.setattr(worker.notifier, "send_alert", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.requests, "post", lambda *a, **k: None)

    calls = []

    class _FakeWriter:
        def __init__(self, *a, **k):
            pass
        def write(self, frame):
            pass
        def release(self):
            pass

    def fake_video_writer(path, fourcc, fps, size):
        calls.append(fps)
        return _FakeWriter()

    monkeypatch.setattr(pipeline_mod.cv2, "VideoWriter", fake_video_writer)
    return calls


def test_save_clip_uses_the_real_achieved_frame_rate(tmp_path, monkeypatch):
    worker = _make_worker(tmp_path)   # cfg.FPS = 15 (see _make_worker)
    calls = _capture_writer_fps(monkeypatch, worker)

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(60)]
    worker._save_clip(frames, elapsed_seconds=30.0)   # 60 frames over 30s real time = 2 fps

    assert len(calls) == 1
    assert calls[0] == 2.0
    assert calls[0] != worker.cfg.FPS   # the bug this regresses: it used to always be 15


def test_save_clip_never_writes_a_zero_or_negative_fps(tmp_path, monkeypatch):
    # A handful of frames over a long real window (the exact scenario
    # that produced "0:00" clips) must still floor at a valid, positive
    # writer fps rather than crashing or producing an unplayable file.
    worker = _make_worker(tmp_path)
    calls = _capture_writer_fps(monkeypatch, worker)

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    worker._save_clip(frames, elapsed_seconds=45.0)   # 3 frames / 45s real time = 0.067 fps

    assert calls[0] >= 1.0


def test_save_clip_falls_back_to_configured_fps_when_elapsed_is_missing(tmp_path, monkeypatch):
    worker = _make_worker(tmp_path)
    calls = _capture_writer_fps(monkeypatch, worker)

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
    worker._save_clip(frames, elapsed_seconds=None)

    assert calls[0] == worker.cfg.FPS


def test_save_clip_falls_back_to_configured_fps_when_elapsed_is_too_small_to_trust(tmp_path, monkeypatch):
    # Guards the div-by-a-near-zero-number edge case (a clock hiccup, or
    # a direct/synchronous caller not going through _start_recording).
    worker = _make_worker(tmp_path)
    calls = _capture_writer_fps(monkeypatch, worker)

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
    worker._save_clip(frames, elapsed_seconds=0.3)

    assert calls[0] == worker.cfg.FPS


def test_start_recording_snapshots_the_pre_event_buffers_real_span(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)

    clock["t"] = 1_000_000.0
    worker.buffer.append(np.zeros((4, 4, 3), dtype=np.uint8))
    worker.buffer_times.append(clock["t"])
    clock["t"] = 1_000_004.0
    worker.buffer.append(np.zeros((4, 4, 3), dtype=np.uint8))
    worker.buffer_times.append(clock["t"])

    clock["t"] = 1_000_010.0
    worker._start_recording(clock["t"])

    # The oldest frame currently in the buffer was appended 10s before
    # this call, not BUFFER_SECONDS (cfg.BUFFER_SECONDS=2 in
    # _make_worker) -- the whole point of tracking real append times
    # instead of assuming the configured window was actually achieved.
    assert worker._record_pre_event_span == 10.0


def test_start_recording_pre_event_span_is_zero_when_buffer_is_empty(tmp_path, monkeypatch):
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)

    worker._start_recording(clock["t"])

    assert worker._record_pre_event_span == 0.0


def test_process_frame_passes_the_real_elapsed_span_to_save_clip(tmp_path, monkeypatch):
    # End-to-end: process_frame's own finalize block must compute and
    # pass elapsed_seconds, not leave _save_clip to guess.
    clock = _fake_clock(monkeypatch)
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(worker, "_dispatch_alert", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    worker.cfg.POST_EVENT_SECONDS = 5

    captured = []
    monkeypatch.setattr(worker, "_save_clip", lambda frames, elapsed_seconds=None, confidence=None: captured.append(elapsed_seconds))

    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[_hazard_event()])
    assert worker.recording is True

    clock["t"] += 5   # exactly POST_EVENT_SECONDS later
    worker.process_frame(_varied_frame(), detections=[], motion_ok=False, hazard_events=[])

    assert len(captured) == 1
    # No pre-event buffer frames were seeded in this test (empty
    # buffer/buffer_times), so elapsed_seconds should be just the
    # post-event span: 5s.
    assert captured[0] == 5.0


# ---------------------------------------------------------------------
# _dispatch_alert's Incident History POST: regression coverage for a
# real production bug. requests.post defaults to verifying the server
# cert against the public CA bundle, which a self-signed cert (the kind
# generated for this app's local HTTPS) can never pass -- so once
# DASHBOARD_URL pointed at https://, every internal alert POST was
# silently failing its SSL handshake, every single time, with the
# failure swallowed by a bare `except Exception: pass` and nothing
# printed. Alerts appeared to work (printed to the console, email
# attempted) but never once reached Incident History. These tests run
# _dispatch_alert for real (with requests.post and notifier faked out,
# and threading.Thread replaced with something that runs synchronously
# so the test does not need to sleep/poll for a background thread).
# ---------------------------------------------------------------------

class _ImmediateThread:
    """Stands in for threading.Thread: runs target(*args) synchronously
    on .start() instead of on a background thread, so a test can assert
    on a dispatched call's effects (e.g. _dispatch_alert, _save_clip)
    immediately without sleeping/polling for a real background thread."""
    def __init__(self, target=None, args=(), daemon=None):
        self._target = target
        self._args   = args

    def start(self):
        self._target(*self._args)


def test_dispatch_alert_verifies_against_the_configured_cert_path(tmp_path, monkeypatch):
    worker = _make_worker(tmp_path)
    worker.cfg.DASHBOARD_CERT_PATH = "/fake/path/to/cert.pem"
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    monkeypatch.setattr(worker.notifier, "send_alert", lambda *a, **k: None)

    calls = []
    def fake_post(url, json=None, headers=None, timeout=None, verify=None):
        calls.append({"url": url, "verify": verify})
    monkeypatch.setattr(pipeline_mod.requests, "post", fake_post)

    worker._dispatch_alert(
        ("CAM-X", "Room X", "Fighting Detected", 0.9, "Check camera immediately"),
        {"event_type": "Fighting Detected"},
    )

    assert len(calls) == 1
    assert calls[0]["verify"] == "/fake/path/to/cert.pem"


def test_dispatch_alert_verify_defaults_true_with_no_cert_path(tmp_path, monkeypatch):
    # Plain-HTTP deployments (no cert generated) must not pass a falsy
    # verify value that would silently disable TLS verification --
    # `verify` should fall back to True (a no-op for a non-https URL,
    # and the safe default if DASHBOARD_URL were ever https anyway).
    worker = _make_worker(tmp_path)
    assert getattr(worker.cfg, "DASHBOARD_CERT_PATH", None) is None
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    monkeypatch.setattr(worker.notifier, "send_alert", lambda *a, **k: None)

    calls = []
    def fake_post(url, json=None, headers=None, timeout=None, verify=None):
        calls.append({"verify": verify})
    monkeypatch.setattr(pipeline_mod.requests, "post", fake_post)

    worker._dispatch_alert(
        ("CAM-X", "Room X", "Fighting Detected", 0.9, "Check camera immediately"),
        {"event_type": "Fighting Detected"},
    )

    assert calls[0]["verify"] is True


def test_dispatch_alert_prints_on_post_failure(tmp_path, monkeypatch, capsys):
    # Regression guard for the silent `except Exception: pass` that let
    # this whole class of bug (SSL verification failures, and before
    # that the DASHBOARD_URL scheme mismatch) go undetected: a failed
    # Incident History POST must now print something, not disappear.
    worker = _make_worker(tmp_path)
    monkeypatch.setattr(pipeline_mod, "threading", SimpleNamespace(Thread=_ImmediateThread))
    monkeypatch.setattr(worker.notifier, "send_alert", lambda *a, **k: None)

    def failing_post(*a, **k):
        raise ConnectionError("simulated SSL/connection failure")
    monkeypatch.setattr(pipeline_mod.requests, "post", failing_post)

    worker._dispatch_alert(
        ("CAM-X", "Room X", "Fighting Detected", 0.9, "Check camera immediately"),
        {"event_type": "Fighting Detected"},
    )

    out = capsys.readouterr().out
    assert "CAM-X" in out
    assert "Incident history POST failed" in out
