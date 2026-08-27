# tests/test_main.py
#
# Unit tests for src/main.py's _build_camera_config, the function that
# turns one cameras.json record into a per-camera detection.config.Config
# used by MultiCameraEngine. This is where a camera's hazard_enabled
# field (added in the dashboard's camera add/edit UI) becomes that
# camera's HAZARD_DETECTION_ENABLED, the flag that decides whether
# MultiCameraEngine._run_cycle fires hazard events for it. If this
# mapping is wrong, the per-camera hazard opt-in/opt-out logic tested in
# test_multi_camera.py never actually gets exercised in production,
# since the engine only ever sees what this function hands it.
#
#   PYTHONPATH=src pytest tests/test_main.py -v

import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

import main  # noqa: E402
from main import _build_camera_config  # noqa: E402


def test_build_camera_config_maps_core_fields():
    cfg = _build_camera_config({
        "id": "CAM-01", "room": "Sunroom Wing", "source": "0", "threshold": 0.85,
    })
    assert cfg.CAMERA_ID == "CAM-01"
    assert cfg.ROOM_NAME == "Sunroom Wing"
    assert cfg.CAMERA_SOURCE == "0"
    assert cfg.VIOLENCE_THRESHOLD == 0.85


def test_build_camera_config_uses_config_default_when_no_dashboard_url_given():
    # Uses main.Config (bound at collection time) rather than a fresh
    # `from detection.config import Config` here: test_dashboard.py's
    # app_client fixture swaps sys.modules["detection.config"] for a
    # shadow module without DASHBOARD_URL, and does not restore it after
    # itself, so a late import in test-execution order (after that
    # fixture has run at least once) would resolve to the shadow, not
    # the real Config -- unrelated to what this test is actually
    # checking.
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert cfg.DASHBOARD_URL == main.Config.DASHBOARD_URL


def test_build_camera_config_applies_dashboard_url_when_given():
    """Regression test: CameraWorker._dispatch_alert POSTs every fired
    alert to f"{DASHBOARD_URL}/api/events/add". Config's class default is
    plain http://, but once a TLS cert is present the dashboard serves
    HTTPS-only on that port and a plain-http POST fails to connect,
    silently (the failure is caught and swallowed on purpose, so a
    dashboard hiccup never crashes detection) -- so every alert fires but
    none of them ever reach Incident History, with nothing in the console
    to say so. main.py must thread the actually-served scheme through to
    every camera's Config, not rely on Config's static default."""
    cfg = _build_camera_config(
        {"id": "CAM-01", "room": "A", "source": "0"},
        dashboard_url="https://localhost:5000",
    )
    assert cfg.DASHBOARD_URL == "https://localhost:5000"


def test_build_camera_config_has_no_cert_path_by_default():
    # A camera config built with no dashboard_cert_path (e.g. plain HTTP,
    # no cert generated) must not claim to have one -- CameraWorker reads
    # this via getattr(..., None), so simply not setting the attribute is
    # correct, not setting it to some placeholder value.
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert getattr(cfg, "DASHBOARD_CERT_PATH", None) is None


def test_build_camera_config_applies_dashboard_cert_path_when_given():
    """Regression test: CameraWorker._dispatch_alert's requests.post
    defaults to verifying the server cert against the public CA bundle,
    which a self-signed cert can never pass -- so once DASHBOARD_URL is
    https://, every internal alert POST silently failed its SSL
    handshake (caught and swallowed, so a dashboard hiccup never crashes
    detection), meaning alerts fired and printed to the console but never
    once reached Incident History. main.py must thread the exact cert
    file the dashboard is serving through to every camera's Config, so
    _dispatch_alert can verify against precisely that known cert."""
    cfg = _build_camera_config(
        {"id": "CAM-01", "room": "A", "source": "0"},
        dashboard_cert_path="/path/to/cert.pem",
    )
    assert cfg.DASHBOARD_CERT_PATH == "/path/to/cert.pem"


def test_build_camera_config_hazard_enabled_defaults_false():
    # A camera record from before this field existed, or one where the
    # caregiver never touched the hazard checkbox, must not silently
    # turn hazard detection on.
    cfg = _build_camera_config({"id": "CAM-01", "room": "Sunroom Wing", "source": "0"})
    assert cfg.HAZARD_DETECTION_ENABLED is False


def test_build_camera_config_hazard_enabled_true_is_passed_through():
    cfg = _build_camera_config({
        "id": "CAM-01", "room": "Sunroom Wing", "source": "0", "hazard_enabled": True,
    })
    assert cfg.HAZARD_DETECTION_ENABLED is True


def test_build_camera_config_picks_up_system_settings(monkeypatch):
    """Regression test: System Settings (confirm_seconds/motion_threshold/
    buffer_seconds/post_event_seconds), edited from the dashboard and
    persisted to system_settings.json, used to have no effect on a
    running engine -- load_system_settings() was written but never
    called from anywhere in the detection startup path. This checks
    _build_camera_config actually applies whatever it returns."""
    monkeypatch.setattr(main, "load_system_settings", lambda: {
        "confirm_seconds": 7,
        "motion_threshold": 2.25,
        "buffer_seconds": 20,
        "post_event_seconds": 42,
    })
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert cfg.CONFIRM_SECONDS == 7
    assert cfg.MOTION_THRESHOLD == 2.25
    assert cfg.BUFFER_SECONDS == 20
    assert cfg.POST_EVENT_SECONDS == 42


def test_build_camera_config_falls_back_to_config_defaults_when_setting_missing(monkeypatch):
    # A partial/older system_settings.json (or load_system_settings()
    # simply not having a key yet) must not crash or null out a field --
    # each field falls back to the Config class default individually.
    from detection.config import Config
    monkeypatch.setattr(main, "load_system_settings", lambda: {"confirm_seconds": 9})
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert cfg.CONFIRM_SECONDS == 9
    assert cfg.MOTION_THRESHOLD == Config.MOTION_THRESHOLD
    assert cfg.BUFFER_SECONDS == Config.BUFFER_SECONDS
    assert cfg.POST_EVENT_SECONDS == Config.POST_EVENT_SECONDS


def test_build_camera_config_picks_up_hazard_supervision_settings(monkeypatch):
    # Same regression as test_build_camera_config_picks_up_system_settings
    # above, for the two hazard supervision-context settings added this
    # week (previously config.py-only, no System Settings control at
    # all -- see dashboard/app.py's _system_settings_with_defaults).
    monkeypatch.setattr(main, "load_system_settings", lambda: {
        "hazard_require_unsupervised": False,
        "hazard_quiet_hours_start": 22,
        "hazard_quiet_hours_end": 6,
    })
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert cfg.HAZARD_REQUIRE_UNSUPERVISED is False
    assert cfg.HAZARD_QUIET_HOURS_START == 22
    assert cfg.HAZARD_QUIET_HOURS_END == 6


def test_build_camera_config_hazard_quiet_hours_null_means_off(monkeypatch):
    # Quiet hours' "off" state is an explicit None, not an absent key --
    # this must survive the round trip through sys_settings.get(), not
    # get coerced to some other falsy value or silently fall back to a
    # stale Config-class default of a *different* value.
    monkeypatch.setattr(main, "load_system_settings", lambda: {
        "hazard_require_unsupervised": True,
        "hazard_quiet_hours_start": None,
        "hazard_quiet_hours_end": None,
    })
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert cfg.HAZARD_QUIET_HOURS_START is None
    assert cfg.HAZARD_QUIET_HOURS_END is None


def test_build_camera_config_falls_back_to_hazard_defaults_when_setting_missing():
    # No System Settings override saved yet (a fresh install, or one
    # from before this feature existed) -- must not crash, and must fall
    # back to whatever config.py itself specifies, same fallback pattern
    # as confirm_seconds/motion_threshold above.
    from detection.config import Config
    cfg = _build_camera_config({"id": "CAM-01", "room": "A", "source": "0"})
    assert cfg.HAZARD_REQUIRE_UNSUPERVISED == getattr(Config, "HAZARD_REQUIRE_UNSUPERVISED", True)
    assert cfg.HAZARD_QUIET_HOURS_START == getattr(Config, "HAZARD_QUIET_HOURS_START", None)
    assert cfg.HAZARD_QUIET_HOURS_END == getattr(Config, "HAZARD_QUIET_HOURS_END", None)


def test_build_camera_config_hazard_enabled_is_per_camera():
    # The whole point of wiring this through: two cameras built from two
    # different records must not share a HAZARD_DETECTION_ENABLED value.
    cfg_on  = _build_camera_config({"id": "CAM-ON", "room": "A", "source": "0", "hazard_enabled": True})
    cfg_off = _build_camera_config({"id": "CAM-OFF", "room": "B", "source": "0", "hazard_enabled": False})
    assert cfg_on.HAZARD_DETECTION_ENABLED is True
    assert cfg_off.HAZARD_DETECTION_ENABLED is False


# ---------------------------------------------------------------------
# start_engine: an inactive camera must not get a worker started for it
# at all, since dashboard/app.py's get_cameras() derives "Live"/"Offline"
# purely from whether a worker is registered for that camera id. These
# tests fake out MultiCameraEngine (via monkeypatch) so they exercise
# only the active-filtering logic, not real model loading.
# ---------------------------------------------------------------------

class _FakeEngine:
    """Stands in for MultiCameraEngine: records what configs it was built
    with instead of loading a real model, and exposes the same
    workers/start() surface start_engine() depends on."""
    def __init__(self, configs):
        if not configs:
            raise ValueError("MultiCameraEngine needs at least one camera config")
        self.configs = configs
        self.workers = {c.CAMERA_ID: object() for c in configs}

    def start(self):
        return self


def test_start_engine_skips_inactive_cameras(monkeypatch):
    monkeypatch.setattr(main, "MultiCameraEngine", _FakeEngine)
    monkeypatch.setattr(main, "register_worker", lambda cam_id, worker: None)

    records = [
        {"id": "CAM-ON",  "room": "A", "source": "0", "active": True},
        {"id": "CAM-OFF", "room": "B", "source": "0", "active": False},
    ]
    engine = main.start_engine(records)
    assert engine is not None
    assert set(engine.workers.keys()) == {"CAM-ON"}


def test_start_engine_defaults_active_true_when_field_missing(monkeypatch):
    # A camera record from before the active flag existed, or one where
    # a client never sent it, must still get started, matching the same
    # default add_camera() uses when persisting a new record.
    monkeypatch.setattr(main, "MultiCameraEngine", _FakeEngine)
    monkeypatch.setattr(main, "register_worker", lambda cam_id, worker: None)
    engine = main.start_engine([{"id": "CAM-X", "room": "A", "source": "0"}])
    assert engine is not None
    assert set(engine.workers.keys()) == {"CAM-X"}


def test_start_engine_returns_none_when_every_camera_is_inactive(monkeypatch):
    monkeypatch.setattr(main, "MultiCameraEngine", _FakeEngine)
    records = [{"id": "CAM-OFF", "room": "B", "source": "0", "active": False}]
    assert main.start_engine(records) is None
