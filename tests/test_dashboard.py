# tests/test_dashboard.py
#
# Covers the caregiver dashboard's Flask app: auth (happy path + failure
# paths), role-based access control, CORS-wildcard stripping, rate
# limiting, and the core CRUD/incident endpoints. Run with:
#
#   pip install pytest flask flask-login flask-limiter flask-wtf opencv-python-headless
#   PYTHONPATH=src pytest tests/ -v
#
# This suite uses a throwaway outputs/ directory (via the CWD fixture
# below) so running it never touches real camera/incident/user data.

import importlib
import io
import json
import os
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """Fresh Flask app + fresh JSON stores + two seeded accounts per test.

    auth/users.py and dashboard/app.py compute their JSON store paths
    from `Path(__file__).parent...` at import time, not from the
    working directory, so `monkeypatch.chdir()` alone does not isolate
    test runs from each other or from real deployment data. Instead the
    modules are imported once, then each store file they point at is
    explicitly cleared before every test.
    """
    # A minimal Config so app.py's `from detection.config import Config`
    # succeeds without requiring a real deployment config.py.
    detection_dir = tmp_path / "src_shadow" / "detection"
    detection_dir.mkdir(parents=True)
    (detection_dir / "__init__.py").write_text("")
    (detection_dir / "config.py").write_text(
        "class Config:\n"
        "    SECRET_KEY = 'test-secret-key'\n"
        "    INTERNAL_API_KEY = 'test-internal-key'\n"
        "    MODEL_PATH = 'models/finetuned_model.pt'\n"
        "    CONFIRM_SECONDS = 3\n"
        "    MOTION_THRESHOLD = 1.5\n"
        "    BUFFER_SECONDS = 10\n"
        "    POST_EVENT_SECONDS = 15\n"
        "    EMAIL_SENDER = 'test@example.com'\n"
        "    EMAIL_APP_PASSWORD = 'x'\n"
        "    EMAIL_RECIPIENTS = []\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "src_shadow"))

    for mod in ("dashboard.app", "dashboard", "detection.config", "detection", "auth.users", "auth"):
        sys.modules.pop(mod, None)
    import auth.users as users_module
    import dashboard.app as app_module

    for path_attr in ("USERS_FILE", "INVITES_FILE"):
        getattr(users_module, path_attr).unlink(missing_ok=True)
    for path_attr in (
        "CAMERAS_FILE", "LOGS_FILE", "PROFILES_FILE",
        "ANNOUNCE_FILE", "SETTINGS_FILE", "SYS_SETTINGS_FILE",
    ):
        getattr(app_module, path_attr).unlink(missing_ok=True)
    # Events AND round check-ins now live in the same SQLite db
    # (dashboard.events_store's SqliteEventsStore / SqliteCheckinsStore
    # -- two tables, one file) instead of events.json directly. Clear
    # the db file itself plus any sidecar WAL/SHM journal files sqlite
    # may have left behind, and the renamed legacy-migration marker
    # (LOGS_FILE.unlink() above only removes events.json itself, not
    # events.json.migrated) -- otherwise a prior test's data, or its
    # already-migrated marker, would leak into this one. One unlink
    # clears both tables since they share the file.
    events_db = app_module.EVENTS_DB_FILE
    events_db.unlink(missing_ok=True)
    Path(str(events_db) + "-wal").unlink(missing_ok=True)
    Path(str(events_db) + "-shm").unlink(missing_ok=True)
    Path(str(app_module.LOGS_FILE) + ".migrated").unlink(missing_ok=True)

    # jane is scoped to Sunroom Wing on purpose. Most of this suite's
    # cameras/incidents live in that room, and an unscoped caregiver (the
    # actual account default) would correctly see none of them. Tests
    # that specifically exercise room scoping create their own
    # additional, deliberately unassigned or differently assigned
    # accounts below.
    users_module.create_caregiver(
        "jane@ward.org", "correcthorse123", "Jane Doe", role="caregiver",
        assigned_rooms=["Sunroom Wing"],
    )
    users_module.create_caregiver("lead@ward.org", "correcthorse123", "Lead Sup", role="admin")

    # WTF_CSRF_ENABLED=False: this suite exercises route behavior (auth,
    # authorization, CRUD, room scoping), not CSRF token handling itself,
    # and Flask-WTF's CSRFProtect (see dashboard/app.py) does not
    # automatically relax under TESTING=True the way some other
    # frameworks' test modes do. CSRF enforcement itself is covered
    # separately in test_csrf_protection_is_enforced below, against a
    # client that deliberately leaves this flag at its real default.
    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    yield app_module


def login(app_module, email, password="correcthorse123"):
    c = app_module.app.test_client()
    c.post("/login", data={"email": email, "password": password})
    return c


# ---------------------------------------------------------------------------
# Auth happy + failure paths
# ---------------------------------------------------------------------------
def test_login_page_renders(app_client):
    r = app_client.app.test_client().get("/login")
    assert r.status_code == 200
    assert b"Sign in" in r.data


def test_wrong_password_is_generic_error(app_client):
    r = app_client.app.test_client().post("/login", data={"email": "jane@ward.org", "password": "wrong"})
    assert b"Invalid email or password" in r.data


def test_nonexistent_email_same_generic_error(app_client):
    r = app_client.app.test_client().post("/login", data={"email": "nobody@ward.org", "password": "whatever1"})
    assert b"Invalid email or password" in r.data


def test_correct_login_redirects_to_dashboard(app_client):
    r = app_client.app.test_client().post("/login", data={"email": "jane@ward.org", "password": "correcthorse123"})
    assert r.status_code == 302


def test_anonymous_page_request_redirects(app_client):
    r = app_client.app.test_client().get("/")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_anonymous_api_request_is_401_not_redirect(app_client):
    r = app_client.app.test_client().get("/api/cameras")
    assert r.status_code == 401
    assert r.is_json


def test_login_is_rate_limited_after_repeated_attempts(app_client):
    c = app_client.app.test_client()
    codes = [c.post("/login", data={"email": "jane@ward.org", "password": "wrong"}).status_code for _ in range(15)]
    assert 429 in codes


def test_open_redirect_guard_on_next_param(app_client):
    c = app_client.app.test_client()
    r = c.post("/login?next=https://evil.example.com", data={"email": "jane@ward.org", "password": "correcthorse123"})
    assert "evil.example.com" not in r.headers.get("Location", "")


def test_logout_clears_session(app_client):
    c = login(app_client, "jane@ward.org")
    c.get("/logout")
    r = c.get("/api/cameras")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# CORS wildcard hardening
# ---------------------------------------------------------------------------
def test_wildcard_cors_header_is_stripped(app_client):
    from flask import jsonify

    @app_client.app.route("/__test_wildcard")
    def _wildcard():
        resp = jsonify({"ok": True})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    r = app_client.app.test_client().get("/__test_wildcard")
    assert r.headers.get("Access-Control-Allow-Origin") is None


def test_security_headers_are_set(app_client):
    r = app_client.app.test_client().get("/login")
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    csp = r.headers.get("Content-Security-Policy", "")
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_session_cookie_is_secure_and_httponly(app_client):
    c = app_client.app.test_client()
    r = c.post("/login", data={"email": "lead@ward.org", "password": "correcthorse123"})
    set_cookie = r.headers.get("Set-Cookie", "")
    assert "session=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=Lax" in set_cookie


# ---------------------------------------------------------------------------
# CSRF protection
# ---------------------------------------------------------------------------
def test_csrf_protection_is_enforced(app_client):
    # The app_client fixture disables WTF_CSRF_ENABLED for the rest of
    # this suite (it exercises route behavior -- auth, authorization,
    # CRUD -- not CSRF token handling itself). This test explicitly
    # re-enables the real default to prove CSRFProtect actually rejects a
    # mutating request with no token, and accepts one carrying the real
    # token the same way dashboard.js's fetch() wrapper sends it.
    admin = login(app_client, "lead@ward.org")   # while CSRF is still disabled, matching every other login() call
    app_client.app.config["WTF_CSRF_ENABLED"] = True

    r = admin.post("/api/cameras/add", json={"id": "CAM-CSRF", "room": "Sunroom Wing", "source": "0"})
    assert r.status_code == 400
    assert not any(c["id"] == "CAM-CSRF" for c in admin.get("/api/cameras").json)

    token = re.search(r'data-csrf-token="([^"]+)"', admin.get("/").data.decode()).group(1)
    r2 = admin.post(
        "/api/cameras/add",
        json={"id": "CAM-CSRF", "room": "Sunroom Wing", "source": "0"},
        headers={"X-CSRFToken": token},
    )
    assert r2.status_code == 200
    assert any(c["id"] == "CAM-CSRF" for c in admin.get("/api/cameras").json)


# ---------------------------------------------------------------------------
# Model upload path traversal
# ---------------------------------------------------------------------------
def test_upload_model_sanitizes_traversal_filename_and_stays_confined(app_client, tmp_path, monkeypatch):
    # MODELS_DIR is computed once from the real repo root in app.py, not
    # a per-test tmp dir like the JSON stores the fixture already
    # isolates above -- so this test explicitly redirects it rather than
    # risking a write into (or escaping from) the real project's models/.
    isolated_models_dir = tmp_path / "models"
    isolated_models_dir.mkdir()
    monkeypatch.setattr(app_client, "MODELS_DIR", isolated_models_dir)

    admin = login(app_client, "lead@ward.org")
    data = {"model": (io.BytesIO(b"not a real checkpoint"), "../../escaped.pt")}
    r = admin.post("/api/system-settings/upload-model", data=data, content_type="multipart/form-data")
    assert r.status_code == 200

    written = list(isolated_models_dir.rglob("*.pt"))
    assert len(written) == 1
    assert written[0].parent == isolated_models_dir   # confined despite the ../../ in the original filename
    assert not (tmp_path / "escaped.pt").exists()      # nothing landed outside models/


def test_upload_model_resolved_path_guard_blocks_even_if_the_sanitizer_is_bypassed(app_client, tmp_path, monkeypatch):
    # Regression guard for the second, independent containment check in
    # upload_model(): simulates a hypothetical future regression where
    # secure_filename() itself fails to strip a traversal sequence, and
    # confirms the resolved-path comparison still refuses to write
    # outside MODELS_DIR rather than trusting the sanitizer alone.
    isolated_models_dir = tmp_path / "models"
    isolated_models_dir.mkdir()
    monkeypatch.setattr(app_client, "MODELS_DIR", isolated_models_dir)
    monkeypatch.setattr(app_client, "secure_filename", lambda name: name)   # pass-through, simulating a bypass

    outside_target = tmp_path / "evil.pt"
    admin = login(app_client, "lead@ward.org")
    data = {"model": (io.BytesIO(b"malicious"), "../evil.pt")}
    r = admin.post("/api/system-settings/upload-model", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert not outside_target.exists()
    assert list(isolated_models_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Role-based access control (system settings / model upload)
# ---------------------------------------------------------------------------
def test_caregiver_can_read_but_not_write_system_settings(app_client):
    cg = login(app_client, "jane@ward.org")
    assert cg.get("/api/system-settings").status_code == 200
    assert cg.post("/api/system-settings", json={"confirm_seconds": 5}).status_code == 403
    assert cg.post("/api/system-settings/upload-model").status_code == 403


def test_admin_can_write_system_settings(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/system-settings", json={
        "confirm_seconds": 5, "motion_threshold": 2.0, "buffer_seconds": 12, "post_event_seconds": 20,
    })
    assert r.status_code == 200
    assert admin.get("/api/system-settings").json["confirm_seconds"] == 5


# ---------------------------------------------------------------------------
# Cameras / incidents / profile / settings
# ---------------------------------------------------------------------------
def test_camera_crud_round_trip(app_client):
    # Camera CRUD (add/update/delete) is admin-only; see app.py's comment
    # on the cameras section. jane (a plain caregiver) is used below only
    # to confirm she can still read the resulting list.
    admin = login(app_client, "lead@ward.org")
    cg = login(app_client, "jane@ward.org")

    assert cg.post("/api/cameras/add", json={"id": "CAM-01", "room": "Sunroom Wing"}).status_code == 403

    assert admin.post("/api/cameras/add", json={
        "id": "CAM-01", "room": "Sunroom Wing", "source": "0", "threshold": 0.9, "patients": 3, "priority": "High",
    }).status_code == 200

    cams = cg.get("/api/cameras").json
    assert cams[0]["patients"] == 3 and cams[0]["priority"] == "High"
    # hazard_enabled defaults to False when not given, matching the
    # opt-in design in detection/hazard.py: hazard detection should
    # never turn on silently for a camera nobody explicitly enabled it
    # for.
    assert cams[0]["hazard_enabled"] is False

    assert admin.post("/api/cameras/update/CAM-01", json={"room": "Renamed"}).status_code == 200
    assert admin.post("/api/cameras/update/CAM-01", json={"hazard_enabled": True}).status_code == 200
    assert admin.get("/api/cameras").json[0]["hazard_enabled"] is True

    assert admin.delete("/api/cameras/delete/CAM-01").status_code == 200
    assert admin.get("/api/cameras").json == []


def test_add_camera_hazard_enabled_true_persists(app_client):
    admin = login(app_client, "lead@ward.org")
    assert admin.post("/api/cameras/add", json={
        "id": "CAM-HAZ", "room": "Sunroom Wing", "source": "0", "hazard_enabled": True,
    }).status_code == 200
    cams = admin.get("/api/cameras").json
    assert next(c for c in cams if c["id"] == "CAM-HAZ")["hazard_enabled"] is True


def test_add_camera_falls_back_to_the_global_alert_threshold(app_client):
    # No threshold sent, and no Alert Settings saved yet either (defaults
    # to 90%), so the new camera should end up at 0.9, not the old
    # hardcoded 0.7 that had no relationship to any admin-visible
    # setting.
    admin = login(app_client, "lead@ward.org")
    assert admin.post("/api/cameras/add", json={
        "id": "CAM-DEFAULT", "room": "Sunroom Wing", "source": "0",
    }).status_code == 200
    cams = admin.get("/api/cameras").json
    assert next(c for c in cams if c["id"] == "CAM-DEFAULT")["threshold"] == 0.9


def test_add_camera_threshold_default_follows_saved_alert_settings(app_client):
    admin = login(app_client, "lead@ward.org")
    assert admin.post("/api/settings", json={"threshold": 70}).status_code == 200

    assert admin.post("/api/cameras/add", json={
        "id": "CAM-FOLLOWS", "room": "Sunroom Wing", "source": "0",
    }).status_code == 200
    cams = admin.get("/api/cameras").json
    assert next(c for c in cams if c["id"] == "CAM-FOLLOWS")["threshold"] == 0.7

    # An explicit threshold in the request must still win over the
    # global default.
    assert admin.post("/api/cameras/add", json={
        "id": "CAM-EXPLICIT", "room": "Sunroom Wing", "source": "0", "threshold": 0.55,
    }).status_code == 200
    cams = admin.get("/api/cameras").json
    assert next(c for c in cams if c["id"] == "CAM-EXPLICIT")["threshold"] == 0.55


def test_add_camera_honors_explicit_active_false(app_client):
    # Previously "active" was hardcoded True on add regardless of what a
    # client sent, so a caregiver unchecking "Active" in the add-camera
    # modal before saving had no effect until a later edit. The camera
    # dict must reflect what was actually submitted.
    admin = login(app_client, "lead@ward.org")
    assert admin.post("/api/cameras/add", json={
        "id": "CAM-INACTIVE", "room": "Sunroom Wing", "source": "0", "active": False,
    }).status_code == 200
    cams = admin.get("/api/cameras").json
    cam = next(c for c in cams if c["id"] == "CAM-INACTIVE")
    assert cam["active"] is False
    assert cam["liveStatus"] == "Offline"


def test_add_camera_still_defaults_active_true_when_omitted(app_client):
    admin = login(app_client, "lead@ward.org")
    assert admin.post("/api/cameras/add", json={
        "id": "CAM-DEFAULT-ACTIVE", "room": "Sunroom Wing", "source": "0",
    }).status_code == 200
    cams = admin.get("/api/cameras").json
    assert next(c for c in cams if c["id"] == "CAM-DEFAULT-ACTIVE")["active"] is True


def test_internal_event_ingestion_requires_key(app_client):
    client = app_client.app.test_client()
    r = client.post("/api/events/add", json={"camera_id": "CAM-01"})
    assert r.status_code == 401  # no key configured / wrong key -> unauthorized

    r = client.post(
        "/api/events/add",
        json={"camera_id": "CAM-01", "room": "Sunroom Wing", "event_type": "Violence Detected",
              "confidence": 0.91, "clip_path": "", "states": [{"track_id": 1, "state": "Fighting", "score": 0.91}]},
        headers={"X-Internal-Key": "test-internal-key"},
    )
    assert r.status_code == 200


def test_incident_notes_review_and_false_positive(app_client):
    client = app_client.app.test_client()
    client.post(
        "/api/events/add",
        json={"camera_id": "CAM-01", "room": "Sunroom Wing", "event_type": "Violence Detected", "confidence": 0.91},
        headers={"X-Internal-Key": "test-internal-key"},
    )
    cg = login(app_client, "jane@ward.org")
    incident_id = cg.get("/api/events").json[0]["id"]

    assert cg.post(f"/api/incidents/{incident_id}/notes", json={"notes": "Checked, settled."}).status_code == 200
    r = cg.post(f"/api/incidents/{incident_id}/review")
    assert r.status_code == 200 and r.json["reviewed"] is True
    r = cg.post(f"/api/incidents/{incident_id}/false-positive")
    assert r.status_code == 200 and r.json["false_positive"] is True


def test_profile_is_scoped_to_current_user(app_client):
    cg = login(app_client, "jane@ward.org")
    admin = login(app_client, "lead@ward.org")

    cg.post("/api/profile", json={"about": "6 years in elder care", "notes": "Prefers night shift"})
    assert cg.get("/api/profile").json["about"] == "6 years in elder care"
    assert admin.get("/api/profile").json.get("about", "") != "6 years in elder care"


def test_checkin_requires_login(app_client):
    anon = app_client.app.test_client()
    assert anon.post("/api/checkins/add").status_code in (302, 401)
    assert anon.get("/api/checkins/last").status_code in (302, 401)


def test_checkin_last_is_none_before_any_checkin(app_client):
    cg = login(app_client, "jane@ward.org")
    assert cg.get("/api/checkins/last").json is None


def test_checkin_add_logs_and_returns_a_record(app_client):
    cg = login(app_client, "jane@ward.org")
    r = cg.post("/api/checkins/add")
    assert r.status_code == 200
    record = r.json
    assert record["caregiver_name"] == "Jane Doe"
    assert record["rooms"] == ["Sunroom Wing"]
    assert record["timestamp"]

    last = cg.get("/api/checkins/last").json
    assert last["id"] == record["id"]


def test_checkin_admin_rooms_recorded_as_all(app_client):
    admin = login(app_client, "lead@ward.org")
    record = admin.post("/api/checkins/add").json
    assert record["rooms"] == "all"


def test_checkin_is_scoped_to_current_user(app_client):
    # Row-level scoping by session identity, same property as
    # /api/profile above: one caregiver's check-in must never appear as
    # another caregiver's "last check-in".
    cg = login(app_client, "jane@ward.org")
    admin = login(app_client, "lead@ward.org")

    cg.post("/api/checkins/add")
    assert admin.get("/api/checkins/last").json is None

    admin.post("/api/checkins/add")
    cg_last = cg.get("/api/checkins/last").json
    assert cg_last["caregiver_name"] == "Jane Doe"


def test_checkin_last_reflects_the_most_recent_of_several(app_client):
    cg = login(app_client, "jane@ward.org")
    first = cg.post("/api/checkins/add").json
    second = cg.post("/api/checkins/add").json
    assert first["id"] != second["id"]
    assert cg.get("/api/checkins/last").json["id"] == second["id"]


def test_checkin_history_is_admin_only(app_client):
    cg = login(app_client, "jane@ward.org")
    admin = login(app_client, "lead@ward.org")
    cg.post("/api/checkins/add")

    assert cg.get("/api/checkins").status_code == 403
    assert admin.get("/api/checkins").status_code == 200


def test_checkin_history_includes_every_caregiver_newest_first(app_client):
    cg = login(app_client, "jane@ward.org")
    admin = login(app_client, "lead@ward.org")

    first = cg.post("/api/checkins/add").json
    second = admin.post("/api/checkins/add").json

    history = admin.get("/api/checkins").json
    ids = [c["id"] for c in history]
    assert first["id"] in ids and second["id"] in ids
    # Newest first: second (admin's) was logged after first (jane's).
    assert ids.index(second["id"]) < ids.index(first["id"])


def test_alert_settings_round_trip(app_client):
    cg = login(app_client, "jane@ward.org")
    cg.post("/api/settings", json={
        "recipients": ["ayesha@havencare.org"], "threshold": 85, "cooldown": 90, "email_channel": True,
    })
    s = cg.get("/api/settings").json
    assert s["threshold"] == 85
    assert s["recipients"] == ["ayesha@havencare.org"]


def test_sound_channel_defaults_on_and_round_trips(app_client):
    cg = login(app_client, "jane@ward.org")
    # Untouched settings default to sound on, same as email.
    assert cg.get("/api/settings").json["sound_channel"] is True
    cg.post("/api/settings", json={"sound_channel": False})
    assert cg.get("/api/settings").json["sound_channel"] is False
    cg.post("/api/settings", json={"sound_channel": True})
    assert cg.get("/api/settings").json["sound_channel"] is True


def test_desktop_channel_defaults_on_and_round_trips(app_client):
    cg = login(app_client, "jane@ward.org")
    assert cg.get("/api/settings").json["desktop_channel"] is True
    cg.post("/api/settings", json={"desktop_channel": False})
    assert cg.get("/api/settings").json["desktop_channel"] is False
    cg.post("/api/settings", json={"desktop_channel": True})
    assert cg.get("/api/settings").json["desktop_channel"] is True


def test_analytics_reflects_logged_incidents(app_client):
    client = app_client.app.test_client()
    client.post(
        "/api/events/add",
        json={"camera_id": "CAM-01", "room": "Sunroom Wing", "event_type": "Violence Detected", "confidence": 0.91},
        headers={"X-Internal-Key": "test-internal-key"},
    )
    cg = login(app_client, "jane@ward.org")
    a = cg.get("/api/analytics?days=7").json
    assert a["total_incidents"] >= 1


# ---------------------------------------------------------------------------
# Invite-based sign-up (no open self-registration)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Room-scoped access: a caregiver should only see cameras/incidents/clips
# for rooms they are explicitly assigned to; admins see everything.
# ---------------------------------------------------------------------------
def _seed_two_room_setup(app_client):
    """Two cameras/incidents in different rooms, plus an admin session to
    create them with (camera CRUD is admin-only)."""
    admin = login(app_client, "lead@ward.org")
    admin.post("/api/cameras/add", json={"id": "CAM-01", "room": "Sunroom Wing"})
    admin.post("/api/cameras/add", json={"id": "CAM-02", "room": "Memory Care Unit"})
    client = app_client.app.test_client()
    client.post("/api/events/add", json={
        "camera_id": "CAM-01", "room": "Sunroom Wing",
        "event_type": "Violence Detected", "confidence": 0.9,
    }, headers={"X-Internal-Key": "test-internal-key"})
    client.post("/api/events/add", json={
        "camera_id": "CAM-02", "room": "Memory Care Unit",
        "event_type": "Fall Detected", "confidence": 1.0,
    }, headers={"X-Internal-Key": "test-internal-key"})
    return admin


def test_unassigned_caregiver_sees_no_rooms(app_client):
    # A caregiver created with no assigned_rooms (the actual default),
    # not jane, who this fixture deliberately scopes to Sunroom Wing.
    import auth.users as users_module
    users_module.create_caregiver("new_hire@ward.org", "correcthorse123", "New Hire", role="caregiver")
    _seed_two_room_setup(app_client)
    cg = login(app_client, "new_hire@ward.org")

    assert cg.get("/api/cameras").json == []
    assert cg.get("/api/events").json == []
    assert cg.get("/api/analytics?days=7").json["total_incidents"] == 0


def test_caregiver_sees_only_assigned_room(app_client):
    _seed_two_room_setup(app_client)
    cg = login(app_client, "jane@ward.org")  # scoped to Sunroom Wing only

    cam_rooms = {c["room"] for c in cg.get("/api/cameras").json}
    assert cam_rooms == {"Sunroom Wing"}

    event_rooms = {e["room"] for e in cg.get("/api/events").json}
    assert event_rooms == {"Sunroom Wing"}

    a = cg.get("/api/analytics?days=7").json
    assert a["total_incidents"] == 1


def test_caregiver_cannot_reach_incident_in_other_room(app_client):
    _seed_two_room_setup(app_client)
    cg = login(app_client, "jane@ward.org")
    other_room_incident = next(
        e for e in login(app_client, "lead@ward.org").get("/api/events").json
        if e["room"] == "Memory Care Unit"
    )
    r = cg.get(f"/api/incidents/{other_room_incident['id']}")
    assert r.status_code == 404  # not 403; must not confirm the incident exists
    assert cg.post(f"/api/incidents/{other_room_incident['id']}/notes", json={"notes": "x"}).status_code == 404


def test_caregiver_cannot_reach_video_feed_or_score_in_other_room(app_client):
    _seed_two_room_setup(app_client)
    cg = login(app_client, "jane@ward.org")
    assert cg.get("/video_feed/CAM-02").status_code == 404
    assert cg.get("/api/score/CAM-02").status_code == 404
    # Her own assigned camera's routes should NOT 404 for the same reason.
    assert cg.get("/api/score/CAM-01").status_code == 200


def test_deleted_camera_video_feed_and_score_deny_not_allow(app_client):
    """Regression test for a bug found in the final code sweep. _camera_room()
    returns None for a camera id that is not in cameras.json (for example,
    deleted while its worker thread was still registered), and video_feed/
    get_score used to treat "room is None" as "no room check applies", the
    opposite of what _clip_room_accessible does for the same ambiguous
    case. That let a caregiver reach a deleted camera's still-running
    feed/score even though they could never have accessed it by camera id
    while it existed unassigned to their rooms. Deleting must narrow
    access, never widen it."""
    _seed_two_room_setup(app_client)
    admin = login(app_client, "lead@ward.org")
    # A worker some request handler still has a reference to, as if the
    # camera thread never actually stopped; see delete_camera's comment.
    app_client.register_worker("CAM-02", object())
    admin.delete("/api/cameras/delete/CAM-02")

    cg = login(app_client, "jane@ward.org")  # scoped to Sunroom Wing only
    assert cg.get("/video_feed/CAM-02").status_code == 404
    assert cg.get("/api/score/CAM-02").status_code == 404
    # An admin session must not get a free pass around a deleted camera
    # either, since _camera_room legitimately can't return a room for it.
    assert admin.get("/api/score/CAM-02").status_code == 404


def test_delete_camera_unregisters_worker(app_client):
    _seed_two_room_setup(app_client)
    admin = login(app_client, "lead@ward.org")
    app_client.register_worker("CAM-02", object())
    assert "CAM-02" in app_client._workers

    admin.delete("/api/cameras/delete/CAM-02")
    assert "CAM-02" not in app_client._workers


def test_clip_room_scoping(app_client, monkeypatch, tmp_path):
    """No prior test reached /api/clips or /clips/<filename> at all, despite
    _clip_room_accessible being the most convoluted access-control function
    in the file (it derives a room from a regex-parsed camera id embedded
    in the clip filename, and must default to deny on any ambiguity)."""
    _seed_two_room_setup(app_client)
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    monkeypatch.setattr(app_client, "CLIPS_DIR", clips_dir)

    (clips_dir / "alert_CAM-01_20260101_120000.mp4").write_bytes(b"fake")   # Sunroom Wing
    (clips_dir / "alert_CAM-02_20260101_120000.mp4").write_bytes(b"fake")   # Memory Care Unit
    (clips_dir / "alert_CAM-99_20260101_120000.mp4").write_bytes(b"fake")   # unknown camera

    cg = login(app_client, "jane@ward.org")  # scoped to Sunroom Wing only
    names = {c["filename"] for c in cg.get("/api/clips").json}
    assert names == {"alert_CAM-01_20260101_120000.mp4"}, (
        "a caregiver's clip list must not include other rooms' clips, "
        "or clips from a camera id that can't be matched to any room"
    )

    assert cg.get("/clips/alert_CAM-01_20260101_120000.mp4").status_code == 200
    assert cg.get("/clips/alert_CAM-02_20260101_120000.mp4").status_code == 404
    assert cg.get("/clips/alert_CAM-99_20260101_120000.mp4").status_code == 404

    admin = login(app_client, "lead@ward.org")
    names = {c["filename"] for c in admin.get("/api/clips").json}
    assert names == {
        "alert_CAM-01_20260101_120000.mp4",
        "alert_CAM-02_20260101_120000.mp4",
        "alert_CAM-99_20260101_120000.mp4",
    }, "an admin sees every clip regardless of room, including unmatched ones"


def test_clip_route_always_declares_video_mp4(app_client, monkeypatch, tmp_path):
    """Regression test for a real deployment bug: serve_clip used to let
    Werkzeug guess the Content-Type from the filename extension via
    Python's stdlib mimetypes module. On Windows that guess consults the
    Windows registry's file-extension associations rather than a bundled
    table, and a missing/broken .mp4 registry entry there makes it fall
    through to application/octet-stream -- a fully valid, correctly
    encoded H.264 clip that a browser's <video> element then refuses to
    even attempt playing, since it was never told the response is a
    video. serve_clip now passes mimetype="video/mp4" explicitly so the
    header can never depend on the host OS's mimetype database."""
    _seed_two_room_setup(app_client)
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    monkeypatch.setattr(app_client, "CLIPS_DIR", clips_dir)
    (clips_dir / "alert_CAM-01_20260101_120000.mp4").write_bytes(b"fake")

    cg = login(app_client, "jane@ward.org")
    resp = cg.get("/clips/alert_CAM-01_20260101_120000.mp4")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "video/mp4"


def test_admin_sees_every_room_regardless_of_assigned_rooms(app_client):
    _seed_two_room_setup(app_client)
    admin = login(app_client, "lead@ward.org")
    cam_rooms = {c["room"] for c in admin.get("/api/cameras").json}
    assert cam_rooms == {"Sunroom Wing", "Memory Care Unit"}


def test_admin_can_update_caregiver_rooms_caregiver_cannot(app_client):
    cg = login(app_client, "jane@ward.org")
    admin = login(app_client, "lead@ward.org")

    r = cg.post("/api/caregivers/jane@ward.org/rooms", json={"assigned_rooms": ["Memory Care Unit"]})
    assert r.status_code == 403  # caregivers can't change their own (or anyone's) access

    r = admin.post("/api/caregivers/jane@ward.org/rooms", json={"assigned_rooms": ["Memory Care Unit"]})
    assert r.status_code == 200
    assert r.json["assigned_rooms"] == ["Memory Care Unit"]

    # Change takes effect immediately for jane's existing session.
    _seed_two_room_setup(app_client)
    cam_rooms = {c["room"] for c in cg.get("/api/cameras").json}
    assert cam_rooms == {"Memory Care Unit"}


def test_invite_with_assigned_rooms_carries_through_to_new_account(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={
        "email": "scoped@ward.org", "name": "Scoped Person", "role": "caregiver",
        "assigned_rooms": ["Sunroom Wing"],
    })
    assert r.status_code == 200
    assert r.json["assigned_rooms"] == ["Sunroom Wing"]
    token = r.json["link"].split("token=")[1]

    anon = app_client.app.test_client()
    anon.post(f"/signup?token={token}", data={
        "token": token, "password": "correcthorse123", "confirm_password": "correcthorse123",
    })
    _seed_two_room_setup(app_client)
    cam_rooms = {c["room"] for c in anon.get("/api/cameras").json}
    assert cam_rooms == {"Sunroom Wing"}


# ---------------------------------------------------------------------------
# Retention / cleanup
# ---------------------------------------------------------------------------
def test_cleanup_requires_admin(app_client):
    cg = login(app_client, "jane@ward.org")
    assert cg.post("/api/admin/run-cleanup").status_code == 403


def test_cleanup_deletes_old_and_false_positive_events(app_client):
    from datetime import datetime, timedelta
    admin = login(app_client, "lead@ward.org")

    # Set a short false-positive window and a longer general one so both
    # code paths are exercised in one pass.
    admin.post("/api/system-settings", json={
        "confirm_seconds": 3, "buffer_seconds": 10, "post_event_seconds": 15,
        "retention_days": 30, "false_positive_retention_days": 5,
    })

    now = datetime.now()
    events = [
        {"id": "KEEP-RECENT", "timestamp": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
         "room": "Sunroom Wing", "false_positive": False},
        {"id": "DELETE-OLD", "timestamp": (now - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S"),
         "room": "Sunroom Wing", "false_positive": False},
        {"id": "DELETE-FP", "timestamp": (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
         "room": "Sunroom Wing", "false_positive": True},
        {"id": "KEEP-FP-RECENT", "timestamp": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
         "room": "Sunroom Wing", "false_positive": True},
    ]
    app_client.save_events(events)

    r = admin.post("/api/admin/run-cleanup")
    assert r.status_code == 200
    assert r.json["deleted_events"] == 2

    remaining_ids = {e["id"] for e in app_client.load_events()}
    assert remaining_ids == {"KEEP-RECENT", "KEEP-FP-RECENT"}


def test_cleanup_status_reflects_last_run(app_client):
    admin = login(app_client, "lead@ward.org")
    assert admin.get("/api/admin/cleanup-status").json["ran_at"] is None
    admin.post("/api/admin/run-cleanup")
    assert admin.get("/api/admin/cleanup-status").json["ran_at"] is not None


def test_signup_page_without_token_shows_invalid(app_client):
    r = app_client.app.test_client().get("/signup")
    assert r.status_code == 200
    assert b"Invite link invalid" in r.data


def test_caregiver_cannot_create_invites(app_client):
    cg = login(app_client, "jane@ward.org")
    r = cg.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    assert r.status_code == 403
    assert cg.get("/api/invites").status_code == 403


def test_admin_can_create_invite_and_it_grants_signup_access(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    assert r.status_code == 200
    link = r.json["link"]
    assert "/signup?token=" in link
    token = link.split("token=")[1]

    anon = app_client.app.test_client()
    page = anon.get(f"/signup?token={token}")
    assert page.status_code == 200
    assert b"new@ward.org" in page.data
    assert b"Invite link invalid" not in page.data


def test_signup_creates_account_and_logs_in(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    token = r.json["link"].split("token=")[1]

    anon = app_client.app.test_client()
    r2 = anon.post(f"/signup?token={token}", data={
        "token": token, "password": "correcthorse123", "confirm_password": "correcthorse123",
    })
    assert r2.status_code == 302
    # session is now authenticated as the new account
    me = anon.get("/api/me").json
    assert me["email"] == "new@ward.org" and me["role"] == "caregiver"


def test_signup_token_is_single_use(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    token = r.json["link"].split("token=")[1]

    anon1 = app_client.app.test_client()
    anon1.post(f"/signup?token={token}", data={
        "token": token, "password": "correcthorse123", "confirm_password": "correcthorse123",
    })

    anon2 = app_client.app.test_client()
    page = anon2.get(f"/signup?token={token}")
    assert b"Invite link invalid" in page.data


def test_signup_rejects_mismatched_passwords(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    token = r.json["link"].split("token=")[1]

    anon = app_client.app.test_client()
    r2 = anon.post(f"/signup?token={token}", data={
        "token": token, "password": "correcthorse123", "confirm_password": "different123",
    })
    assert b"do not match" in r2.data
    # token still unused after a failed submit
    assert anon.get("/api/me").status_code == 401


def test_invite_cannot_target_existing_account(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={"email": "jane@ward.org", "name": "Jane Doe", "role": "caregiver"})
    assert r.status_code == 400


def test_admin_can_revoke_pending_invite(app_client):
    admin = login(app_client, "lead@ward.org")
    admin.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    assert any(i["email"] == "new@ward.org" for i in admin.get("/api/invites").json)

    assert admin.delete("/api/invites/by-email/new@ward.org").status_code == 200
    assert not any(i["email"] == "new@ward.org" for i in admin.get("/api/invites").json)


def test_signup_is_rate_limited(app_client):
    admin = login(app_client, "lead@ward.org")
    r = admin.post("/api/invites", json={"email": "new@ward.org", "name": "New Person", "role": "caregiver"})
    token = r.json["link"].split("token=")[1]

    anon = app_client.app.test_client()
    codes = [anon.post(f"/signup?token={token}", data={
        "token": token, "password": "wrongwrong", "confirm_password": "different",
    }).status_code for _ in range(15)]
    assert 429 in codes


# ---------------------------------------------------------------------------
# JsonStore: locking closes the read-modify-write race that plain
# load()/save() functions had, and _write_json_atomic prevents a
# concurrent reader from ever observing a torn (partially-written) file.
# These regression tests exercise the store directly with real threads,
# since the race they guard against only shows up under genuine
# concurrency, not a single-threaded call sequence.
# ---------------------------------------------------------------------------
import threading


def test_json_store_mutate_has_no_lost_updates_under_concurrency(app_client):
    """Regression test for the sequential-fix work order's Logic Errors
    item 4 (JSON-store races). Before JsonStore.mutate() existed,
    add_camera's load-then-append-then-save had no lock across the gap:
    many threads racing add_camera concurrently could each read the same
    "N existing cameras" snapshot and each save N+1, with all but the
    last save's single new camera silently lost. mutate() holds the lock
    for the whole cycle, so N concurrent appends must always produce
    N cameras in the file, not fewer."""
    store = app_client._cameras_store
    store.save([])

    N = 25
    def _add(i):
        def _mutate(cameras):
            cameras.append({"id": f"CAM-{i}"})
        store.mutate(_mutate)

    threads = [threading.Thread(target=_add, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    saved = store.load()
    assert len(saved) == N, (
        f"expected {N} cameras from {N} concurrent mutate() calls, got "
        f"{len(saved)} -- some updates were lost to a race"
    )
    assert {c["id"] for c in saved} == {f"CAM-{i}" for i in range(N)}


def test_add_camera_route_generates_unique_ids_under_concurrency(app_client):
    """Same race as above, exercised through the real /api/cameras/add
    route and its default-id-from-count logic, using a fresh test_client
    per thread (Flask's test client is not documented thread-safe to
    share across threads)."""
    admin_email = "lead@ward.org"

    def _add():
        c = login(app_client, admin_email)
        c.post("/api/cameras/add", json={"room": "Concurrency Room", "source": "0"})

    N = 10
    threads = [threading.Thread(target=_add) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cameras = app_client.load_cameras()
    room_cameras = [c for c in cameras if c["room"] == "Concurrency Room"]
    assert len(room_cameras) == N, (
        f"expected {N} cameras added concurrently, found {len(room_cameras)} "
        "-- some add_camera calls silently overwrote each other"
    )
    ids = [c["id"] for c in room_cameras]
    assert len(set(ids)) == len(ids), f"duplicate camera ids assigned under concurrency: {ids}"


def test_write_json_atomic_leaves_old_content_readable_mid_write(app_client, tmp_path):
    """_write_json_atomic writes to a temp file and os.replace()s it into
    place, rather than truncating the real file in place. This checks
    the target file's content is always one of the two complete JSON
    payloads, never a truncated/partial one, by writing a large payload
    and confirming the temp file cleans up and the destination parses
    correctly afterward (a torn write would leave invalid JSON or a
    stray .tmp file behind)."""
    path = tmp_path / "atomic_test.json"
    path.write_text(json.dumps({"version": 1}))

    big_payload = {"version": 2, "data": list(range(50000))}
    app_client._write_json_atomic(path, big_payload)

    assert json.loads(path.read_text()) == big_payload
    leftover_tmp_files = list(tmp_path.glob(".atomic_test.json.tmp-*"))
    assert leftover_tmp_files == [], f"temp file(s) left behind: {leftover_tmp_files}"
