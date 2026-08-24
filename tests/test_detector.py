# tests/test_detector.py
#
# Regression test for the sequential-fix work order's Logic Errors item 7:
# PersonDetector.detect() used to silently drop every person-class box
# when the tracker returned no `ids` array for a frame (a real, expected
# occurrence -- see the comment in detector.py) with zero visibility. This
# checks the now-added rate-limited diagnostic actually fires, and only
# fires once within its rate-limit window.
#
#   PYTHONPATH=src pytest tests/test_detector.py -v

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from detection import detector as detector_mod  # noqa: E402
from detection.detector import PersonDetector  # noqa: E402


class FakeBoxes:
    def __init__(self, xyxy, cls, conf, ids):
        self._xyxy = np.array(xyxy, dtype=float)
        self._cls  = np.array(cls, dtype=int)
        self._conf = np.array(conf, dtype=float)
        self._ids  = None if ids is None else np.array(ids, dtype=int)

    class _NumpyWrap:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def astype(self, dtype):
            return self._arr.astype(dtype)

    @property
    def xyxy(self):
        return FakeBoxes._NumpyWrap(self._xyxy)

    @property
    def cls(self):
        return FakeBoxes._NumpyWrap(self._cls)

    @property
    def conf(self):
        return FakeBoxes._NumpyWrap(self._conf)

    @property
    def id(self):
        return None if self._ids is None else FakeBoxes._NumpyWrap(self._ids)


def make_detector():
    """Bypasses __init__ (which loads a real YOLO model) the same way
    tests/test_pipeline.py does, then fills in just the attributes
    detect() actually reads."""
    d = PersonDetector.__new__(PersonDetector)
    d.conf   = 0.4
    d.device = "cpu"
    d._last_dropped_ids_log = 0.0
    return d


def make_frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def stub_track_result(monkeypatch, detector, boxes):
    fake_result = SimpleNamespace(boxes=boxes)
    detector.model = SimpleNamespace(track=lambda *a, **k: [fake_result])


def test_dropped_ids_logs_once_and_is_rate_limited(monkeypatch, capsys):
    detector = make_detector()
    boxes = FakeBoxes(
        xyxy=[[10, 10, 60, 90]], cls=[0], conf=[0.9], ids=None,
    )
    stub_track_result(monkeypatch, detector, boxes)

    fake_now = [1000.0]
    monkeypatch.setattr(detector_mod.time, "time", lambda: fake_now[0])

    result = detector.detect(make_frame())
    assert result == []  # the box is still correctly dropped
    out = capsys.readouterr().out
    assert "dropping person detection" in out

    # A second frame moments later, still no ids: should NOT log again,
    # it's rate-limited.
    fake_now[0] += 1
    detector.detect(make_frame())
    out2 = capsys.readouterr().out
    assert "dropping person detection" not in out2

    # After the rate-limit window elapses, it logs again.
    fake_now[0] += 11
    detector.detect(make_frame())
    out3 = capsys.readouterr().out
    assert "dropping person detection" in out3


def test_no_log_when_ids_are_present(monkeypatch, capsys):
    detector = make_detector()
    boxes = FakeBoxes(
        xyxy=[[10, 10, 60, 90]], cls=[0], conf=[0.9], ids=[1],
    )
    stub_track_result(monkeypatch, detector, boxes)

    result = detector.detect(make_frame())
    assert len(result) == 1
    out = capsys.readouterr().out
    assert "dropping person detection" not in out


def test_no_log_when_no_person_class_boxes_at_all(monkeypatch, capsys):
    """A frame with only non-person classes (or none) and no ids array
    should not trigger the diagnostic -- there is nothing person-related
    being silently dropped."""
    detector = make_detector()
    boxes = FakeBoxes(
        xyxy=[[10, 10, 60, 90]], cls=[41], conf=[0.9], ids=None,  # class 41 = cup, not tracked here
    )
    stub_track_result(monkeypatch, detector, boxes)

    detector.detect(make_frame())
    out = capsys.readouterr().out
    assert "dropping person detection" not in out
