# tests/test_multi_camera.py
#
# Tests for the batched multi-camera inference engine
# (detection/multi_camera.py) and the pieces it depends on:
# SimpleIOUTracker and hazard._fire_events (detection/pipeline.py,
# detection/hazard.py).
#
# What's covered directly, with no model involved (pure logic, the parts
# most likely to have an off-by-one or state-bleed bug):
#   - SimpleIOUTracker: id persistence across frames, new ids for new
#     boxes, aging out stale tracks, and the property this class exists
#     for instead of reusing Ultralytics' built-in tracker across
#     cameras: that two independent instances (two cameras) never see
#     each other's track ids.
#   - hazard._fire_events: debounce/streak firing, and the same
#     cross-camera independence property for hazard streak dicts.
#
# What's covered with real models (PersonDetector's YOLO auto-downloads
# on first use; a random-initialized ViolenceClassifier state_dict is
# generated on the fly so this suite never needs the project's actual
# trained weights):
#   - PersonDetector.detect_batch: returns one result per input frame,
#     same order, same shape as detect()'s per-frame return minus the
#     track_id.
#   - MultiCameraEngine: an end-to-end smoke test that two synthetic
#     camera frames run through one full cycle without raising, and
#     that a camera contributing no crops (because it has no motion, in
#     this test) does not get a score. This does not assert on real
#     detection accuracy, since there are no real people in a synthetic
#     frame; it asserts the batching/dispatch plumbing itself holds
#     together end to end.
#
#   PYTHONPATH=src pytest tests/test_multi_camera.py -v

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from detection.pipeline import SimpleIOUTracker, ViolenceClassifier, CameraWorker  # noqa: E402
from detection.hazard import _fire_events  # noqa: E402
from detection.detector import PersonDetector  # noqa: E402
from detection import multi_camera as mc  # noqa: E402


# ---------------------------------------------------------------------
# SimpleIOUTracker
# ---------------------------------------------------------------------

def test_iou_tracker_persists_id_for_slightly_moved_box():
    t = SimpleIOUTracker(iou_threshold=0.3, max_age=10)
    ids1 = t.update([(100, 100, 200, 300)])
    ids2 = t.update([(105, 102, 205, 302)])   # small shift, still high IoU
    assert ids1 == ids2
    assert ids1[0] == 1


def test_iou_tracker_assigns_new_id_for_unrelated_box():
    t = SimpleIOUTracker(iou_threshold=0.3, max_age=10)
    ids1 = t.update([(100, 100, 200, 300)])
    ids2 = t.update([(100, 100, 200, 300), (600, 600, 700, 800)])
    assert ids2[0] == ids1[0]
    assert ids2[1] != ids1[0]


def test_iou_tracker_ages_out_after_max_age_misses():
    t = SimpleIOUTracker(iou_threshold=0.3, max_age=2)
    ids1 = t.update([(100, 100, 200, 300)])
    tid = ids1[0]
    t.update([])   # miss 1
    t.update([])   # miss 2
    t.update([])   # miss 3, exceeds max_age=2, track should be dropped
    ids_after = t.update([(100, 100, 200, 300)])
    # a fresh box in the same place should not reuse the aged-out id
    assert ids_after[0] != tid


def test_iou_tracker_recovers_id_within_max_age():
    t = SimpleIOUTracker(iou_threshold=0.3, max_age=3)
    ids1 = t.update([(100, 100, 200, 300)])
    tid = ids1[0]
    t.update([])   # one miss, within max_age
    ids2 = t.update([(100, 100, 200, 300)])
    assert ids2[0] == tid


def test_iou_tracker_instances_are_independent_across_cameras():
    # This is the property the whole class exists for: two "cameras"
    # (two separate tracker instances) assign ids from their own
    # sequence, never bleeding into each other's, even when fed
    # identical-looking boxes at the same time.
    cam_a = SimpleIOUTracker()
    cam_b = SimpleIOUTracker()
    ids_a = cam_a.update([(0, 0, 50, 50)])
    ids_b = cam_b.update([(0, 0, 50, 50)])
    assert ids_a == [1]
    assert ids_b == [1]   # same id number is fine, they are different cameras
    # advance cam_a a few frames; cam_b's next id must be unaffected
    cam_a.update([(0, 0, 50, 50), (500, 500, 600, 600)])
    ids_b_2 = cam_b.update([(0, 0, 50, 50), (900, 900, 950, 950)])
    assert ids_b_2[0] == 1
    assert ids_b_2[1] == 2   # cam_b's own second id, not influenced by cam_a's count


# ---------------------------------------------------------------------
# hazard._fire_events
# ---------------------------------------------------------------------

def test_fire_events_fires_at_min_consecutive():
    streak = {}
    objs = [("knife", 0.9, (0, 0, 10, 10))]
    wrists = [np.array([5.0, 5.0])]   # inside the object's box -> dist 0
    ev1 = _fire_events(objs, wrists, prox_px=50, min_consecutive=2, streak=streak)
    assert ev1 == []   # first sample only, not yet at threshold
    ev2 = _fire_events(objs, wrists, prox_px=50, min_consecutive=2, streak=streak)
    assert len(ev2) == 1
    assert ev2[0]["object"] == "knife"


def test_fire_events_resets_streak_on_gap():
    streak = {}
    objs = [("knife", 0.9, (0, 0, 10, 10))]
    near = [np.array([5.0, 5.0])]
    far  = [np.array([9999.0, 9999.0])]
    _fire_events(objs, near, prox_px=50, min_consecutive=2, streak=streak)   # streak=1
    _fire_events(objs, far, prox_px=50, min_consecutive=2, streak=streak)    # gap, resets to 0
    ev = _fire_events(objs, near, prox_px=50, min_consecutive=2, streak=streak)  # streak=1 again
    assert ev == []   # would have fired here if the gap had not reset it


def test_fire_events_no_wrists_clears_streak():
    streak = {"knife": 5}
    ev = _fire_events([("knife", 0.9, (0, 0, 10, 10))], [], prox_px=50, min_consecutive=2, streak=streak)
    assert ev == []
    assert streak == {}


def test_fire_events_streak_dicts_are_independent_across_cameras():
    objs = [("knife", 0.9, (0, 0, 10, 10))]
    near = [np.array([5.0, 5.0])]
    streak_cam_a = {}
    streak_cam_b = {}
    _fire_events(objs, near, prox_px=50, min_consecutive=2, streak=streak_cam_a)
    # cam_b has seen nothing yet; its streak dict must be completely
    # untouched by cam_a's progress toward min_consecutive, not sharing
    # state through some accidental global.
    assert streak_cam_b == {}
    ev_b = _fire_events(objs, near, prox_px=50, min_consecutive=2, streak=streak_cam_b)
    assert ev_b == []   # cam_b's own first sample, correctly not fired yet


def test_fire_events_survives_label_flicker_between_samples():
    # Regression test: a real production bug. A general-purpose COCO
    # detector run at the low confidences typical for this feature
    # (0.4-0.6) can flicker between visually similar hazard classes --
    # knife and scissors are both a handheld blade shape -- from one
    # sampled frame to the next, for the same physical object. The
    # streak used to be tracked per-label, so "scissors" reaching count
    # 1 and the very next sample reclassifying as "knife" started a
    # *different* counter at 1 -- neither ever reached min_consecutive,
    # so a genuinely continuous hazard near a wrist never fired an
    # event. The streak must track "some hazard object is near a wrist"
    # as one count, not one count per exact label.
    streak = {}
    near = [np.array([5.0, 5.0])]
    as_scissors = [("scissors", 0.52, (0, 0, 10, 10))]
    as_knife    = [("knife", 0.48, (0, 0, 10, 10))]

    ev1 = _fire_events(as_scissors, near, prox_px=50, min_consecutive=2, streak=streak)
    assert ev1 == []   # first sample, not yet at threshold
    ev2 = _fire_events(as_knife, near, prox_px=50, min_consecutive=2, streak=streak)
    # Second sample reclassified the same physical object under a
    # different label -- must still fire regardless. With one sample
    # each way there's no majority, so the reported label is the
    # confidence tiebreak (see test_fire_events_reports_the_majority_
    # label_not_just_the_trigger_sample for the case with an actual
    # majority): scissors read at 0.52 beats knife read at 0.48.
    assert len(ev2) == 1
    assert ev2[0]["object"] == "scissors"


def test_fire_events_reports_the_majority_label_not_just_the_trigger_sample():
    # Regression test for a real reported bug: an incident report said
    # "Scissors Detected" for an object a caregiver could see was
    # actually a knife. Root cause: _fire_events used to report whatever
    # label won the single sample that happened to cross
    # min_consecutive, not what was seen across the whole debounce
    # window. Here two of three samples correctly read "knife" and only
    # the third (the trigger sample) misreads "scissors" -- the fired
    # event must still say "knife", the majority, not "scissors".
    streak = {}
    near = [np.array([5.0, 5.0])]
    as_knife    = [("knife", 0.55, (0, 0, 10, 10))]
    as_scissors = [("scissors", 0.9, (0, 0, 10, 10))]  # deliberately higher-confidence than the knife reads

    _fire_events(as_knife, near, prox_px=50, min_consecutive=3, streak=streak)
    _fire_events(as_knife, near, prox_px=50, min_consecutive=3, streak=streak)
    ev = _fire_events(as_scissors, near, prox_px=50, min_consecutive=3, streak=streak)
    assert len(ev) == 1
    assert ev[0]["object"] == "knife", (
        "two of three samples read 'knife'; the trigger sample's own "
        "label (even at higher confidence) must not override the majority"
    )


# ---------------------------------------------------------------------
# PersonDetector.detect_batch (real YOLO, auto-downloaded)
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    return PersonDetector(device="cpu")


def test_detect_batch_returns_one_result_per_frame(detector):
    frames = [
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.full((240, 320, 3), 128, dtype=np.uint8),
        np.zeros((240, 320, 3), dtype=np.uint8),
    ]
    results = detector.detect_batch(frames)
    assert len(results) == len(frames)
    for r in results:
        assert isinstance(r, list)   # detect()'s plain-list shape when extra_classes=None


def test_detect_batch_with_extra_classes_returns_tuple_per_frame(detector):
    frames = [np.zeros((240, 320, 3), dtype=np.uint8), np.zeros((240, 320, 3), dtype=np.uint8)]
    results = detector.detect_batch(frames, extra_classes={43: "knife"})
    assert len(results) == 2
    for dets, extra in results:
        assert isinstance(dets, list)
        assert isinstance(extra, list)


# ---------------------------------------------------------------------
# MultiCameraEngine smoke test (real models, dummy/random weights)
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_weights_path(tmp_path_factory):
    """A structurally valid ViolenceClassifier checkpoint with random
    weights: enough to load and run forward(), without needing this
    project's actual trained model file."""
    path = tmp_path_factory.mktemp("weights") / "dummy.pt"
    torch.save(ViolenceClassifier().state_dict(), path)
    return str(path)


def _make_config(dummy_weights_path, cam_id, room):
    from detection.config import Config
    cfg = Config()
    cfg.MODEL_PATH    = dummy_weights_path
    cfg.CAMERA_ID     = cam_id
    cfg.ROOM_NAME     = room
    cfg.CAMERA_SOURCE = 0
    cfg.MOTION_THRESHOLD = 999999.0   # deliberately unreachable; see test below
    cfg.FPS = 15
    cfg.BUFFER_SECONDS = 2
    cfg.CONFIRM_SECONDS = 1
    cfg.DASHBOARD_URL = "http://127.0.0.1:1"   # unroutable on purpose; alert POSTs must not raise
    cfg.EMAIL_SENDER = "test@example.com"
    cfg.EMAIL_APP_PASSWORD = "x"
    cfg.EMAIL_RECIPIENTS = []
    cfg.COOLDOWN_SECONDS = 120
    cfg.VIOLENCE_THRESHOLD = 0.9
    cfg.POST_EVENT_SECONDS = 5
    cfg.CLIPS_DIR = "/tmp/does-not-matter"
    return cfg


def test_engine_runs_one_cycle_across_two_cameras_without_crashing(dummy_weights_path, monkeypatch):
    configs = [
        _make_config(dummy_weights_path, "CAM-A", "Room A"),
        _make_config(dummy_weights_path, "CAM-B", "Room B"),
    ]
    engine = mc.MultiCameraEngine(configs)

    # Bypass real camera hardware/threads: feed each capture slot a
    # single synthetic frame directly instead of calling engine.start(),
    # then run exactly one inference cycle.
    frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_b = np.full((240, 320, 3), 200, dtype=np.uint8)
    engine._captures["CAM-A"]._frame, engine._captures["CAM-A"]._is_new = frame_a, True
    engine._captures["CAM-B"]._frame, engine._captures["CAM-B"]._is_new = frame_b, True

    engine._run_cycle()   # must not raise

    # MOTION_THRESHOLD was set unreachably high above, so motion_ok is
    # False for both cameras on this synthetic (motionless-relative-to-
    # itself, single-frame) input. That means no crops enter the shared
    # classifier batch and no state_machine.update() calls happen this
    # frame, mirroring the original single-camera _predict()'s
    # early-return behavior. Assert that held: no score should have been
    # recorded, and no track state created from a motion-gated frame.
    for cam_id in ("CAM-A", "CAM-B"):
        worker = engine.workers[cam_id]
        assert worker.score == 0.0
        assert worker.state_machine.tracks == {}


def test_engine_workers_share_one_transform_instance(dummy_weights_path):
    """Regression test for the Performance Bottlenecks sequential-fix
    item: each CameraWorker used to build its own functionally-identical
    T.Compose preprocessing pipeline. In shared mode they should all
    reuse the one instance MultiCameraEngine builds once."""
    configs = [
        _make_config(dummy_weights_path, "CAM-A", "Room A"),
        _make_config(dummy_weights_path, "CAM-B", "Room B"),
    ]
    engine = mc.MultiCameraEngine(configs)
    transform_a = engine.workers["CAM-A"].transform
    transform_b = engine.workers["CAM-B"].transform
    assert transform_a is transform_b


def test_engine_second_cycle_with_no_new_frames_is_a_noop(dummy_weights_path):
    configs = [_make_config(dummy_weights_path, "CAM-A", "Room A")]
    engine = mc.MultiCameraEngine(configs)
    # No frame ever set, so _run_cycle should return early, not crash on
    # an empty batch.
    engine._run_cycle()
    assert engine.workers["CAM-A"].score == 0.0


def test_engine_routes_scores_to_the_correct_camera_and_track(dummy_weights_path, monkeypatch):
    """The one test that matters most for this whole feature: with
    multiple people batched across multiple cameras in one classifier
    call, does each resulting score land back on the exact (camera,
    track) it came from, never swapped, never dropped, never given to
    the wrong room. Bypasses real YOLO detection (a blank synthetic frame
    has no real person to detect) by monkeypatching detect_batch to
    return controlled synthetic detections instead, so this test is
    about the engine's own batching/routing arithmetic, not YOLO's
    accuracy."""
    configs = [
        _make_config(dummy_weights_path, "CAM-A", "Room A"),
        _make_config(dummy_weights_path, "CAM-B", "Room B"),
    ]
    for cfg in configs:
        cfg.MOTION_THRESHOLD = -1.0   # deliberately always satisfied this time

    engine = mc.MultiCameraEngine(configs)

    # CAM-A has two people (track ids 1 and 2 from its own tracker),
    # CAM-B has one (track id 1 from its own tracker; same number,
    # different camera, must not collide).
    crop = np.full((64, 64, 3), 100, dtype=np.uint8)

    def fake_detect_batch(frames, extra_classes=None):
        return [
            [((0, 0, 64, 64), crop), ((100, 100, 164, 164), crop)],   # CAM-A: 2 boxes
            [((0, 0, 64, 64), crop)],                                  # CAM-B: 1 box
        ]
    monkeypatch.setattr(engine.detector, "detect_batch", fake_detect_batch)

    frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_b = np.full((240, 320, 3), 200, dtype=np.uint8)
    engine._captures["CAM-A"]._frame, engine._captures["CAM-A"]._is_new = frame_a, True
    engine._captures["CAM-B"]._frame, engine._captures["CAM-B"]._is_new = frame_b, True

    engine._run_cycle()

    # Every track that got a box this cycle must have been updated in
    # its own camera's state machine: CAM-A has exactly 2 tracks, CAM-B
    # exactly 1, and no track_id number collision caused one to
    # overwrite or merge with the other.
    tracks_a = engine.workers["CAM-A"].state_machine.tracks
    tracks_b = engine.workers["CAM-B"].state_machine.tracks
    assert len(tracks_a) == 2
    assert len(tracks_b) == 1
    assert set(tracks_a.keys()) == {1, 2}
    assert set(tracks_b.keys()) == {1}
    # Every track got a real score in [0, 1] from the shared classifier
    # batch, confirming scoring actually happened and was not silently
    # skipped.
    for track in list(tracks_a.values()) + list(tracks_b.values()):
        assert 0.0 <= track.avg_score() <= 1.0


def test_engine_respects_per_camera_hazard_opt_out(dummy_weights_path, monkeypatch):
    """Regression test for a real bug caught in review: with hazard
    detection enabled on one camera and disabled on another in the same
    engine, the opted-out camera must never fire a hazard event just
    because the shared detect_batch call (which is necessarily one call
    across every camera) happened to report a candidate object in its
    frame too. Only the camera that actually opted in may fire."""
    cfg_on  = _make_config(dummy_weights_path, "CAM-ON", "Room On")
    cfg_off = _make_config(dummy_weights_path, "CAM-OFF", "Room Off")
    cfg_on.HAZARD_DETECTION_ENABLED     = True
    cfg_on.HAZARD_MIN_CONSECUTIVE       = 1   # fire on the first sample, keep the test short
    cfg_on.HAZARD_SAMPLE_EVERY_N_FRAMES = 1   # be due on the very first cycle, not the 5th
    cfg_off.HAZARD_DETECTION_ENABLED    = False

    engine = mc.MultiCameraEngine([cfg_on, cfg_off])
    assert engine.hazard_detector is not None   # shared pose model loaded, since CAM-ON opted in

    crop = np.full((64, 64, 3), 100, dtype=np.uint8)
    # Both cameras "detect" an identical knife right next to a person
    # box; the only difference between them is HAZARD_DETECTION_ENABLED.
    def fake_detect_batch(frames, extra_classes=None):
        knife = [("knife", 0.95, (10, 10, 30, 30))]
        return [
            ([((0, 0, 64, 64), crop)], knife),   # CAM-ON
            ([((0, 0, 64, 64), crop)], knife),   # CAM-OFF
        ]
    monkeypatch.setattr(engine.detector, "detect_batch", fake_detect_batch)

    # Wrist sitting right on top of the knife box for every camera in
    # the batch. predict_wrists_batch is monkeypatched instead of
    # running a real pose model, since the point of this test is the
    # per-camera gating logic, not pose accuracy.
    monkeypatch.setattr(
        engine.hazard_detector, "predict_wrists_batch",
        lambda frames: [[np.array([15.0, 15.0])] for _ in frames],
    )

    frame_on  = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_off = np.full((240, 320, 3), 200, dtype=np.uint8)
    engine._captures["CAM-ON"]._frame, engine._captures["CAM-ON"]._is_new = frame_on, True
    engine._captures["CAM-OFF"]._frame, engine._captures["CAM-OFF"]._is_new = frame_off, True

    engine._run_cycle()

    assert engine._hazard_streaks["CAM-ON"].get("_any_hazard_object", 0) >= 1
    # The bug: without the per-camera gate, this would also be >= 1,
    # because the shared detect_batch call reports a candidate object
    # for CAM-OFF too. With the fix, CAM-OFF's streak dict must never
    # even be touched, since it opted out.
    assert engine._hazard_streaks["CAM-OFF"] == {}


def test_engine_resets_hazard_streak_on_a_real_gap(dummy_weights_path, monkeypatch):
    """Regression test for a real bug caught in review: a camera's
    hazard streak must reset to 0 on a due sample where no candidate
    object was seen at all, not just when a DIFFERENT label drops out
    while some other object is still present. Before the fix, a due
    sample with an empty candidate list was skipped entirely (no call
    ever reached the streak-clearing logic), so a knife seen once, then
    genuinely absent for a while, then seen again would silently resume
    counting from its old streak instead of restarting at 1."""
    cfg = _make_config(dummy_weights_path, "CAM-GAP", "Room Gap")
    cfg.HAZARD_DETECTION_ENABLED     = True
    cfg.HAZARD_MIN_CONSECUTIVE       = 2   # needs 2 consecutive due samples to fire
    cfg.HAZARD_SAMPLE_EVERY_N_FRAMES = 1   # every cycle is due

    engine = mc.MultiCameraEngine([cfg])

    crop  = np.full((64, 64, 3), 100, dtype=np.uint8)
    knife = [("knife", 0.95, (10, 10, 30, 30))]
    # Cycle 1: knife present. Cycle 2: nothing detected at all (the real
    # gap). Cycle 3: knife present again -- with the fix this is only
    # the SECOND consecutive sample since the gap, so it must not fire
    # yet. Before the fix, cycle 2 would be skipped entirely and cycle 3
    # would incorrectly be the streak's second hit, firing early.
    cycles = [
        ([((0, 0, 64, 64), crop)], knife),
        ([((0, 0, 64, 64), crop)], []),
        ([((0, 0, 64, 64), crop)], knife),
    ]
    call_count = {"n": 0}

    def fake_detect_batch(frames, extra_classes=None):
        result = cycles[call_count["n"]]
        call_count["n"] += 1
        return [result]
    monkeypatch.setattr(engine.detector, "detect_batch", fake_detect_batch)
    monkeypatch.setattr(
        engine.hazard_detector, "predict_wrists_batch",
        lambda frames: [[np.array([15.0, 15.0])] for _ in frames],
    )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    engine._captures["CAM-GAP"]._frame, engine._captures["CAM-GAP"]._is_new = frame, True
    engine._run_cycle()
    assert engine._hazard_streaks["CAM-GAP"].get("_any_hazard_object", 0) == 1

    engine._captures["CAM-GAP"]._frame, engine._captures["CAM-GAP"]._is_new = frame, True
    engine._run_cycle()
    assert engine._hazard_streaks["CAM-GAP"] == {}   # the gap must clear it

    engine._captures["CAM-GAP"]._frame, engine._captures["CAM-GAP"]._is_new = frame, True
    engine._run_cycle()
    # restarted at 1, not resumed at 2
    assert engine._hazard_streaks["CAM-GAP"].get("_any_hazard_object", 0) == 1


def test_save_clip_uses_the_snapshot_it_was_given_not_a_live_reference(dummy_weights_path, tmp_path):
    """Regression test for the other real issue caught in review:
    _save_clip() is dispatched onto a background thread while
    process_frame() immediately resets self.record_frames to a fresh
    empty list on the calling thread right after. If _save_clip() read
    self.record_frames itself instead of taking an explicit snapshot
    argument, a background thread scheduled even slightly late would see
    the already-reset empty list and silently save an empty/corrupt clip.
    This calls _save_clip directly (not through the async dispatch) to
    isolate exactly the snapshot-vs-live-reference behavior."""
    from detection.pipeline import CameraWorker
    from detection.detector import PersonDetector
    from detection.pipeline import ViolenceClassifier
    import torch as _torch

    cfg = _make_config(dummy_weights_path, "CAM-X", "Room X")
    cfg.CLIPS_DIR = str(tmp_path)
    shared = {
        "model": ViolenceClassifier(),
        "detector": PersonDetector.__new__(PersonDetector),   # not used by _save_clip
        "hazard_detector": None,
        "hazard_class_map": None,
    }
    worker = CameraWorker(cfg, shared=shared)
    # This test is about the snapshot-vs-live-reference behavior, not
    # about actually sending mail. The notifier is stubbed so
    # _save_clip()'s real send_alert() call does not attempt a genuine
    # SMTP connection (which would hang or fail slowly depending on
    # network egress rules, not fail fast the way a broken localhost
    # connection would).
    worker.notifier = SimpleNamespace(send_alert=lambda *a, **k: None)

    real_frames = [np.full((32, 32, 3), i, dtype=np.uint8) for i in range(5)]
    worker.record_frames = real_frames

    # Simulate exactly what process_frame does: snapshot the reference,
    # then reset the live attribute, before the snapshot is ever used.
    frames_to_save = worker.record_frames
    worker.record_frames = []

    worker._save_clip(frames_to_save)   # must save the 5 real frames, not []

    saved = list(tmp_path.glob("*.mp4"))
    assert len(saved) == 1
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(saved[0]))
    frame_count = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 5


# ---------------------------------------------------------------------
# _CaptureThread resilience: regression coverage for a real production
# bug found via a user's console log. cap.read() returning ret=False --
# which happens on a completely ordinary, transient hiccup (a Windows
# MSMF backend glitch, a USB blip, another app briefly grabbing the
# device) -- used to `break` the capture loop and release the camera
# for good. From that moment on, self._is_new never became True again,
# so MultiCameraEngine._run_cycle's "if not frames: return" silently
# did nothing every cycle forever: no detection, no hazard checks, no
# alerts, no live-feed refresh, and nothing printed to say so after the
# very first warning -- indistinguishable from the app just being
# "stuck" rather than visibly broken. These tests exercise
# _CaptureThread.run() directly against a fake cv2.VideoCapture so no
# real camera hardware is needed.
# ---------------------------------------------------------------------

class _FakeCapture:
    """Stands in for cv2.VideoCapture. `reads` is a queue of (ret, frame)
    tuples consumed one per .read() call; once exhausted, keeps
    returning (False, None) so a test can also exercise the reopen
    path without needing to know exactly how many reads happen."""

    instances = []  # class-level: every _FakeCapture constructed, in order

    def __init__(self, src):
        self.src = src
        self.reads = []
        self.released = False
        _FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_capture_thread_survives_a_single_dropped_frame(monkeypatch):
    # One ret=False amid otherwise-good reads must not kill the thread:
    # the next good frame must still come through as a fresh frame.
    # Exercises the real run() logic directly (not a reimplementation of
    # it), driven via a controlled read sequence that signals stop once
    # consumed, so the test does not need a real background thread/join.
    monkeypatch.setattr(mc, "time", SimpleNamespace(sleep=lambda s: None, time=__import__("time").time))
    frame_a = np.full((4, 4, 3), 1, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 2, dtype=np.uint8)

    thread = mc._CaptureThread("CAM-X", 0)

    cap = _FakeCapture(0)
    cap.reads = [(True, frame_a), (False, None), (True, frame_b)]
    monkeypatch.setattr(mc.cv2, "VideoCapture", lambda src: cap)

    def stop_after_reads():
        # Let run() consume exactly the 3 scripted reads, then signal stop
        # so the loop exits instead of spinning on the now-empty queue.
        orig_read = cap.read
        count = {"n": 0}
        def wrapped():
            count["n"] += 1
            ret, frame = orig_read()
            if count["n"] >= 3:
                thread._stop.set()
            return ret, frame
        cap.read = wrapped

    stop_after_reads()
    thread.run()

    frame, is_new = thread.get_latest()
    assert is_new is True
    assert np.array_equal(frame, frame_b)
    assert cap.released is True


def test_capture_thread_reopens_after_sustained_failure(monkeypatch):
    # Enough consecutive failures (not just one) must trigger releasing
    # the dead handle and opening a fresh cv2.VideoCapture, rather than
    # spinning on a handle that never recovers on its own.
    monkeypatch.setattr(mc, "time", SimpleNamespace(sleep=lambda s: None, time=__import__("time").time))

    good_frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    dead_cap  = _FakeCapture(0)
    dead_cap.reads = []   # always (False, None): a genuinely stuck handle
    fresh_cap = _FakeCapture(0)
    fresh_cap.reads = [(True, good_frame)]

    handles = [dead_cap, fresh_cap]
    monkeypatch.setattr(mc.cv2, "VideoCapture", lambda src: handles.pop(0))

    thread = mc._CaptureThread("CAM-X", 0)

    call_count = {"n": 0}
    orig_read = dead_cap.read
    def dead_read():
        call_count["n"] += 1
        if call_count["n"] >= mc._CaptureThread._MAX_CONSECUTIVE_FAILURES_BEFORE_REOPEN:
            thread._stop.set()   # avoid spinning past the reopen point in this test
        return orig_read()
    dead_cap.read = dead_read

    thread.run()

    # The dead handle must have been released once the failure threshold
    # was hit, and a second (fresh) handle opened in its place.
    assert dead_cap.released is True
