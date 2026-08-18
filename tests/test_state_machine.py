# tests/test_state_machine.py
#
# Direct unit tests for the fall-detection rule in
# detection/state_machine.py. This logic has no prior dedicated test --
# it was only exercised indirectly (or not at all) via the Flask-level
# suite in test_dashboard.py, which can't reach the per-person state
# machine at all. That gap is why two real bugs here (a track stuck
# Confirmed forever, and a single noisy frame hard-resetting a Suspected
# fall) were only caught by manual code review, not by tests.
#
# _update_fall is driven by time.time(), not frame count, so these tests
# control a fake clock (monkeypatched into the state_machine module)
# instead of sleeping -- deterministic and fast.
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


def feed(machine, track_id, bbox, clock, dt=0.1):
    """One sample: advance the fake clock, then update height/bbox and
    run the fall rule -- mirrors what StateMachine.update() does per
    frame for a real camera worker."""
    clock.advance(dt)
    track = machine._get_or_create(track_id)
    track.update_bbox(bbox)
    track.update_height(bbox)
    machine._update_fall(track)
    return track


def standing_bbox():
    return (100, 100, 160, 300)  # 60 wide x 200 tall


def fallen_bbox():
    return (100, 250, 300, 310)  # 200 wide x 60 tall -- collapsed + horizontal


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

    # Resume the fall for the rest of the confirm window -- should still
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
