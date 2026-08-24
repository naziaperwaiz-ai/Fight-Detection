# tests/test_state_machine.py
#
# Direct unit tests for the fall-detection rule in
# detection/state_machine.py. This logic had no prior dedicated test;
# it was only exercised indirectly, or not at all, via the Flask-level
# suite in test_dashboard.py, which cannot reach the per-person state
# machine at all. That gap is why two real bugs here (a track stuck
# Confirmed forever, and a single noisy frame hard-resetting a Suspected
# fall) were only caught by manual code review, not by tests.
#
# _update_fall is driven by time.time(), not frame count, so these tests
# control a fake clock (monkeypatched into the state_machine module)
# instead of sleeping, for determinism and speed.
#
#   PYTHONPATH=src pytest tests/test_state_machine.py -v

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from detection import state_machine as sm  # noqa: E402


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, secs):
        self.t += secs


@pytest.fixture()
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(sm.time, "time", fake)
    return fake


def make_machine(clock, **overrides):
    cfg = SimpleNamespace(
        FALL_HEIGHT_DROP_RATIO=0.5,
        FALL_LOOKBACK_SECONDS=2.0,
        FALL_CONFIRM_SECONDS=2.0,
        FALL_MIN_BBOX_HEIGHT=40,
        **overrides,
    )
    return sm.StateMachine(cfg)


def make_violence_machine(clock, **overrides):
    """Like make_machine, but also carries the violence-state-machine
    config (VIOLENCE_THRESHOLD is read directly, not via getattr, so it
    must always be present)."""
    defaults = dict(
        FALL_HEIGHT_DROP_RATIO=0.5,
        FALL_LOOKBACK_SECONDS=2.0,
        FALL_CONFIRM_SECONDS=2.0,
        FALL_MIN_BBOX_HEIGHT=40,
        VIOLENCE_THRESHOLD=0.9,
        CONFIRM_SECONDS=1,
        FPS=15,
    )
    defaults.update(overrides)
    return sm.StateMachine(SimpleNamespace(**defaults))


def feed(machine, track_id, bbox, clock, dt=0.1):
    """One sample: advance the fake clock, then update height/bbox and
    run the fall rule. Mirrors what StateMachine.update() does per frame
    for a real camera worker."""
    clock.advance(dt)
    track = machine._get_or_create(track_id)
    track.update_bbox(bbox)
    track.update_height(bbox)
    machine._update_fall(track)
    return track


def standing_bbox():
    return (100, 100, 160, 300)  # 60 wide x 200 tall


def fallen_bbox():
    return (100, 250, 300, 310)  # 200 wide x 60 tall, collapsed and horizontal


def test_sustained_fall_confirms(clock):
    m = make_machine(clock)
    track = None
    # Establish a standing baseline first, so reference_standing_height > 0.
    for _ in range(10):
        track = feed(m, "p1", standing_bbox(), clock)
    assert track.fall_status == "None"
    # Now collapse + go horizontal for longer than FALL_CONFIRM_SECONDS.
    for _ in range(30):
        track = feed(m, "p1", fallen_bbox(), clock)
    assert track.fall_status == "Confirmed"


def test_brief_occlusion_during_suspected_does_not_reset(clock):
    """Regression test for the bug found in the final sweep: a single
    too-small/noisy bbox sample while Suspected used to hard-reset
    fall_status straight to 'None', discarding all progress toward
    confirmation. One brief occlusion mid-window should not do that --
    it should behave like the already-fixed Confirmed-state recovery
    grace period."""
    m = make_machine(clock)
    for _ in range(10):
        track = feed(m, "p1", standing_bbox(), clock)

    # Collapse + horizontal, but not yet long enough to confirm.
    for _ in range(5):
        track = feed(m, "p1", fallen_bbox(), clock)
    assert track.fall_status == "Suspected"

    # One noisy/occluded sample: bbox height below FALL_MIN_BBOX_HEIGHT,
    # so is_horizontal-based collapse can't be positively read this frame.
    track = feed(m, "p1", (100, 290, 110, 300), clock)  # 10x10, too small
    assert track.fall_status == "Suspected", (
        "a single occluded sample should not discard Suspected progress"
    )

    # Resume the fall for the rest of the confirm window; should still
    # reach Confirmed based on cumulative time since fall_status_since.
    for _ in range(30):
        track = feed(m, "p1", fallen_bbox(), clock)
    assert track.fall_status == "Confirmed"


def test_standing_back_up_from_suspected_resets_to_none(clock):
    """The occlusion tolerance above should not become a loophole that
    never resets: someone who genuinely gets back up should still clear
    back to 'None', just not instantly on one bad frame."""
    m = make_machine(clock)
    for _ in range(10):
        track = feed(m, "p1", standing_bbox(), clock)
    for _ in range(5):
        track = feed(m, "p1", fallen_bbox(), clock)
    assert track.fall_status == "Suspected"

    # Genuinely stands back up and stays that way past the recovery grace
    # window (recovery_secs = 3.0 in the implementation).
    for _ in range(40):
        track = feed(m, "p1", standing_bbox(), clock)
    assert track.fall_status == "None"


def test_confirmed_recovers_after_sustained_non_horizontal(clock):
    """Regression test for the earlier-session bug: a track stuck
    Confirmed forever because a too-small bbox short-circuited the whole
    function. Confirmed should decay back to None after recovery_secs of
    the person genuinely no longer being horizontal."""
    m = make_machine(clock)
    for _ in range(10):
        track = feed(m, "p1", standing_bbox(), clock)
    for _ in range(30):
        track = feed(m, "p1", fallen_bbox(), clock)
    assert track.fall_status == "Confirmed"

    for _ in range(40):
        track = feed(m, "p1", standing_bbox(), clock)
    assert track.fall_status == "None"


# --- Violence state machine: sequential-fix regression tests --------------
#
# These exercise StateMachine.update()/update_all() directly (the real
# per-frame API pipeline.py calls), unlike the fall-only tests above which
# call the private helpers.


def vertical_bbox():
    return (100, 100, 160, 300)   # 60 wide x 200 tall -- standing


def horizontal_bbox():
    return (100, 250, 300, 310)   # 200 wide x 60 tall -- on the ground


def drive_to_state(m, track_id, clock, target_state, dt=0.2):
    """Push a single track from Normal up through the chain to
    target_state by feeding whatever score/bbox each transition needs.
    Returns the track."""
    track = m._get_or_create(track_id)
    order = [sm.PersonState.NORMAL, sm.PersonState.AGITATED,
              sm.PersonState.FIGHTING, sm.PersonState.ON_GROUND,
              sm.PersonState.EMERGENCY]
    target_index = order.index(target_state)

    if target_index >= 1:  # reach AGITATED
        for _ in range(20):
            clock.advance(dt)
            m.update(track_id, 0.9, vertical_bbox(), None)
            if track.state == sm.PersonState.AGITATED:
                break
    if target_index >= 2:  # reach FIGHTING
        for _ in range(20):
            clock.advance(dt)
            m.update(track_id, 0.99, vertical_bbox(), None)
            if track.state == sm.PersonState.FIGHTING:
                break
    if target_index >= 3:  # reach ON_GROUND (needs sustained horizontal)
        for _ in range(10):
            clock.advance(dt)
            m.update(track_id, 0.99, horizontal_bbox(), None)
            if track.state == sm.PersonState.ON_GROUND:
                break
    if target_index >= 4:  # reach EMERGENCY (needs time_in_state > 30s)
        clock.advance(31)
        m.update(track_id, 0.0, horizontal_bbox(), None)
    return track


def test_single_noisy_frame_does_not_flip_fighting_to_on_ground(clock):
    """Regression test: FIGHTING -> OnGround used to key off a single
    frame's is_horizontal() reading. One noisy horizontal frame (e.g. a
    limb crossing the box mid-swing) should not by itself escalate to
    OnGround; it must be sustained for STATE_HORIZONTAL_DEBOUNCE_FRAMES
    consecutive frames."""
    m = make_violence_machine(clock)
    track = drive_to_state(m, "p1", clock, sm.PersonState.FIGHTING)
    assert track.state == sm.PersonState.FIGHTING

    clock.advance(0.2)
    m.update("p1", 0.99, horizontal_bbox(), None)  # single horizontal frame
    assert track.state == sm.PersonState.FIGHTING, (
        "one horizontal frame should not alone trigger OnGround"
    )

    # Back to vertical resets the run counter; still shouldn't have moved.
    clock.advance(0.2)
    m.update("p1", 0.99, vertical_bbox(), None)
    assert track.state == sm.PersonState.FIGHTING


def test_sustained_horizontal_does_flip_fighting_to_on_ground(clock):
    m = make_violence_machine(clock)
    track = drive_to_state(m, "p1", clock, sm.PersonState.FIGHTING)
    assert track.state == sm.PersonState.FIGHTING

    for _ in range(5):
        clock.advance(0.2)
        m.update("p1", 0.99, horizontal_bbox(), None)
    assert track.state == sm.PersonState.ON_GROUND


def test_emergency_recovers_to_normal_after_sustained_recovery(clock):
    """Regression test: EMERGENCY previously had no recovery path at all
    ('stays until manually reset', with nothing in the codebase that ever
    reset it). A person sustained back on their feet past
    STATE_EMERGENCY_RECOVER_SECONDS should clear back to Normal."""
    m = make_violence_machine(clock, STATE_EMERGENCY_RECOVER_SECONDS=2)
    track = drive_to_state(m, "p1", clock, sm.PersonState.EMERGENCY)
    assert track.state == sm.PersonState.EMERGENCY

    # Sustained vertical frames, then enough elapsed time in-state.
    for _ in range(5):
        clock.advance(0.2)
        m.update("p1", 0.0, vertical_bbox(), None)
    clock.advance(3)
    m.update("p1", 0.0, vertical_bbox(), None)
    assert track.state == sm.PersonState.NORMAL


def test_emergency_does_not_recover_on_single_noisy_frame(clock):
    m = make_violence_machine(clock, STATE_EMERGENCY_RECOVER_SECONDS=2)
    track = drive_to_state(m, "p1", clock, sm.PersonState.EMERGENCY)
    assert track.state == sm.PersonState.EMERGENCY

    clock.advance(5)
    m.update("p1", 0.0, vertical_bbox(), None)  # only one vertical frame
    assert track.state == sm.PersonState.EMERGENCY, (
        "a single vertical frame should not alone clear Emergency"
    )


def test_score_window_scales_with_fps(clock):
    """Regression test: avg_score()'s window used to be a hardcoded
    maxlen=15 regardless of FPS. It should now span CONFIRM_SECONDS worth
    of samples at the configured FPS."""
    m = make_violence_machine(clock, CONFIRM_SECONDS=2, FPS=10)
    track = m._get_or_create("p1")
    assert track.score_history.maxlen == 20

    m2 = make_violence_machine(clock, CONFIRM_SECONDS=1, FPS=30)
    track2 = m2._get_or_create("p2")
    assert track2.score_history.maxlen == 30


def test_cleanup_runs_once_per_frame_via_update_all(clock):
    """Regression test: _cleanup_old_tracks() used to run inside update(),
    once per detection per frame, rather than once per frame in
    update_all(). This checks the behavior still works from the real
    per-frame call sequence (per-detection update() calls followed by one
    update_all() call): stale tracks are still dropped after 5s of no
    detections."""
    m = make_violence_machine(clock)
    m.update("stale", 0.1, vertical_bbox(), None)
    m.update_all([("stale", vertical_bbox(), None)])
    assert "stale" in m.tracks

    clock.advance(6)
    m.update("fresh", 0.1, vertical_bbox(), None)
    m.update_all([("fresh", vertical_bbox(), None)])
    assert "stale" not in m.tracks
    assert "fresh" in m.tracks


# --- Motion/proximity backup signal ----------------------------------------
#
# Covers StateMachine._update_motion_fight_pair, added so that real
# violence the classifier under-scores (e.g. a blurry clip averaging
# 0.39, well below STATE_AGITATED_SCORE's 0.4 default) still escalates to
# Fighting -- and therefore still triggers a saved clip, since
# pipeline.py only calls _start_recording from inside the
# has_fighting()/score-threshold alert blocks. All scores fed below stay
# well under STATE_AGITATED_SCORE deliberately, so any escalation to
# Fighting in these tests can only have come from the motion signal, not
# the existing score ladder.

def moving_bbox(step, x0=100, gap=0):
    """60x200 bbox (same footprint as vertical_bbox) whose x position
    advances by 30px per step, offset by `gap` so a second track can be
    kept proximate to the first while both move."""
    x1 = x0 + gap + step * 30
    y1 = 100
    x2 = x1 + 60
    y2 = 300
    return (x1, y1, x2, y2)


def drive_mutual_motion(m, clock, steps, dt=0.1, gap=40, stall_at=None, stall_steps=0):
    """Feed two proximate tracks (p1, p2) moving together, low score, via
    the real per-frame API (update() per detection + one update_all()).
    If stall_at is set, both tracks hold still (no position change) for
    stall_steps steps starting at that step index, then resume moving --
    used to test the brief-interruption recovery tolerance."""
    t1 = t2 = None
    stalled_bbox1 = stalled_bbox2 = None
    for i in range(steps):
        clock.advance(dt)
        if stall_at is not None and stall_at <= i < stall_at + stall_steps:
            if stalled_bbox1 is None:
                stalled_bbox1 = moving_bbox(i, gap=0)
                stalled_bbox2 = moving_bbox(i, gap=gap)
            b1, b2 = stalled_bbox1, stalled_bbox2
        else:
            b1 = moving_bbox(i, gap=0)
            b2 = moving_bbox(i, gap=gap)
        t1 = m.update("p1", 0.1, b1, None)
        t2 = m.update("p2", 0.1, b2, None)
        m.update_all([("p1", b1, None), ("p2", b2, None)])
        if m.tracks["p1"].state == sm.PersonState.FIGHTING:
            break
    return m.tracks["p1"], m.tracks["p2"]


def test_mutual_fast_motion_escalates_to_fighting_despite_low_score(clock):
    """Two proximate tracks both moving fast enough, for long enough,
    should escalate straight to Fighting even though their violence
    score (0.1) never clears STATE_AGITATED_SCORE (0.4 default) -- this
    is the whole point of the backup signal."""
    m = make_violence_machine(clock)
    t1, t2 = drive_mutual_motion(m, clock, steps=40)
    assert t1.state == sm.PersonState.FIGHTING
    assert t2.state == sm.PersonState.FIGHTING
    assert t1.motion_confirmed_fight is True
    assert t2.motion_confirmed_fight is True


def test_single_fast_mover_near_stationary_person_does_not_escalate(clock):
    """A person moving fast next to someone standing still is common
    (someone walking briskly past a stationary caregiver) and must not
    read as a fight -- BOTH tracks must be moving fast, not just one."""
    m = make_violence_machine(clock)
    for i in range(40):
        clock.advance(0.1)
        b1 = moving_bbox(i, gap=0)      # p1 moves
        b2 = vertical_bbox()            # p2 stands still, right next to p1's start
        m.update("p1", 0.1, b1, None)
        m.update("p2", 0.1, b2, None)
        m.update_all([("p1", b1, None), ("p2", b2, None)])
    assert m.tracks["p1"].state != sm.PersonState.FIGHTING
    assert m.tracks["p2"].state != sm.PersonState.FIGHTING


def test_brief_stall_does_not_reset_motion_fight_progress(clock):
    """Regression-style test mirroring _update_fall's occlusion
    tolerance: a couple of still frames mid-scuffle (STATE_MOTION_FIGHT_
    RECOVER_SECONDS's default is 1.0s) should not throw away sustained-
    motion progress built up so far."""
    m = make_violence_machine(clock)
    # Build up most of the confirm window, stall briefly (well under the
    # 1.0s recovery grace period), then resume -- should still confirm.
    t1, t2 = drive_mutual_motion(m, clock, steps=40, stall_at=15, stall_steps=3)
    assert t1.state == sm.PersonState.FIGHTING, (
        "a brief stall well under STATE_MOTION_FIGHT_RECOVER_SECONDS "
        "should not have discarded sustained-motion progress"
    )
    assert t2.state == sm.PersonState.FIGHTING


def test_long_stall_resets_motion_fight_progress(clock):
    """Unlike the brief stall above, a stall longer than
    STATE_MOTION_FIGHT_RECOVER_SECONDS should genuinely reset progress,
    so the pair must sustain fast motion for a fresh confirm window
    afterward rather than picking up where it left off."""
    m = make_violence_machine(clock)
    t1, t2 = drive_mutual_motion(m, clock, steps=12, stall_at=6, stall_steps=15)
    # 15 stalled steps * 0.1s = 1.5s, well past the 1.0s recovery grace
    # period, so progress should have been wiped -- 12 real motion steps
    # total (many of them stalled) is nowhere near a fresh confirm window.
    assert t1.state != sm.PersonState.FIGHTING
    assert t2.state != sm.PersonState.FIGHTING


def test_motion_confirmed_fight_flag_clears_on_recovery(clock):
    """Once a motion-confirmed Fighting track recovers back down the
    ladder (via the normal score-based recovery path), the
    motion_confirmed_fight flag should reset -- otherwise a later,
    unrelated Fighting entry via the score path would be mislabeled as
    motion-confirmed in alert output."""
    m = make_violence_machine(clock)
    t1, t2 = drive_mutual_motion(m, clock, steps=40)
    assert t1.state == sm.PersonState.FIGHTING
    assert t1.motion_confirmed_fight is True

    # Recover via the existing score-based path: low score, sustained,
    # with both tracks staying put (no more fast motion). p2 is moved far
    # away so the proximity check doesn't immediately re-promote p1 from
    # Normal back to Proximate once it recovers -- that would be correct
    # state-machine behavior, just not what this test is isolating.
    far_bbox = (5000, 100, 5060, 300)
    for _ in range(30):
        clock.advance(0.2)
        m.update("p1", 0.0, vertical_bbox(), None)
        m.update("p2", 0.0, far_bbox, None)
        m.update_all([("p1", vertical_bbox(), None), ("p2", far_bbox, None)])
    assert m.tracks["p1"].state == sm.PersonState.NORMAL
    assert m.tracks["p1"].motion_confirmed_fight is False


def test_overlapping_duplicate_boxes_never_escalate_to_fighting(clock):
    """Regression test for a real reported bug: a single person moving
    around ALONE had the live feed unblur and a clip get recorded, with
    no second person anywhere in frame. The likely mechanism is a
    tracker glitch that briefly reports one real body as two track ids
    -- which, before this guard, looked identical to "two proximate
    people both moving fast" to _update_motion_fight_pair. A duplicate
    detection's box overlaps almost entirely with the original (unlike
    two real people, who essentially never do even mid-grapple), so
    feeding the SAME bbox to two track ids must never escalate,
    regardless of how long or how fast the shared motion is."""
    m = make_violence_machine(clock)
    for i in range(60):
        clock.advance(0.1)
        b = moving_bbox(i, gap=0)   # identical box fed to both track ids
        m.update("p1", 0.1, b, None)
        m.update("dup", 0.1, b, None)
        m.update_all([("p1", b, None), ("dup", b, None)])
    assert m.tracks["p1"].state != sm.PersonState.FIGHTING, (
        "two heavily-overlapping boxes should read as one body, not a fight"
    )
    assert m.tracks["dup"].state != sm.PersonState.FIGHTING


def test_freshly_spawned_second_track_cannot_immediately_trigger_motion_fight(clock):
    """A second guard against the same single-person false-positive: a
    track that just appeared hasn't existed long enough to be trusted as
    a genuine second person yet -- a duplicate-detection glitch is, by
    definition, a brand new track id. Feeding a non-overlapping (so
    guard 1 doesn't already cover this) but freshly-spawned second track
    should not be able to escalate p1 within
    STATE_MOTION_FIGHT_MIN_TRACK_AGE_SECONDS of p2 first appearing."""
    m = make_violence_machine(clock)
    for i in range(15):
        clock.advance(0.1)
        b1 = moving_bbox(i, gap=0)
        m.update("p1", 0.1, b1, None)
        m.update_all([("p1", b1, None)])
    assert m.tracks["p1"].state != sm.PersonState.FIGHTING

    # p2 spawns now and is fed 0.5s of motion -- well under the 1.0s
    # default STATE_MOTION_FIGHT_MIN_TRACK_AGE_SECONDS.
    for i in range(15, 20):
        clock.advance(0.1)
        b1 = moving_bbox(i, gap=0)
        b2 = moving_bbox(i, gap=200)   # offset, not overlapping p1
        m.update("p1", 0.1, b1, None)
        m.update("p2", 0.1, b2, None)
        m.update_all([("p1", b1, None), ("p2", b2, None)])
    assert m.tracks["p1"].state != sm.PersonState.FIGHTING, (
        "a track younger than STATE_MOTION_FIGHT_MIN_TRACK_AGE_SECONDS "
        "should not be able to co-trigger the motion-fight signal"
    )
    assert m.tracks["p2"].state != sm.PersonState.FIGHTING
