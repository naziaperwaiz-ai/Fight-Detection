# tests/test_dashboard.py
#
# Covers the caregiver dashboard's Flask app: auth (happy path + failure
# paths), role-based access control, CORS-wildcard stripping, rate
# limiting, and the core CRUD/incident endpoints. Run with:
#
#   pip install pytest flask flask-login flask-limiter opencv-python-headless
#   PYTHONPATH=src pytest tests/ -v
#
# This suite uses a throwaway outputs/ directory (via the CWD fixture
# below) so running it never touches real camera/incident/user data.

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """Fresh Flask app + fresh JSON stores + two seeded accounts per test.

    auth/users.py and dashboard/app.py compute their JSON store paths from
    `Path(__file__).parent...` at import time, not from the working
    directory -- so `monkeypatch.chdir()` alone would NOT isolate test runs
    from each other or from real deployment data. Instead we import the
    modules once, then explicitly clear each store file they point at
    before every test.
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

    # jane is scoped to Sunroom Wing on purpose -- most of this suite's
    # cameras/incidents live in that room, and an unscoped caregiver (the
    # actual account default) would correctly see none of them. Tests that
    # specifically exercise room scoping create their own additional,
    # deliberately-unassigned or differently-assigned accounts below.
    users_module.create_caregiver(
        "jane@ward.org", "correcthorse123", "Jane Doe", role="caregiver",
        assigned_rooms=["Sunroom Wing"],
    )
    users_module.create_caregiver("lead@ward.org", "correcthorse123", "Lead Sup", role="admin")

    app_module.app.config.update(TESTING=True)
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
    # Camera CRUD (add/update/delete) is admin-only -- see app.py's comment
    # on the cameras section. jane (a plain caregiver) is used below only
    # to confirm she can still READ the resulting list.
    admin = login(app_client, "lead@ward.org")
    cg = login(app_client, "jane@ward.org")

    assert cg.post("/api/cameras/add", json={"id": "CAM-01", "room": "Sunroom Wing"}).status_code == 403

    assert admin.post("/api/cameras/add", json={
        "id": "CAM-01", "room": "Sunroom Wing", "source": "0", "threshold": 0.9, "patients": 3, "priority": "High",
    }).status_code == 200

    cams = cg.get("/api/cameras").json
    assert cams[0]["patients"] == 3 and cams[0]["priority"] == "High"

    assert admin.post("/api/cameras/update/CAM-01", json={"room": "Renamed"}).status_code == 200
    assert admin.delete("/api/cameras/delete/CAM-01").status_code == 200
    assert admin.get("/api/cameras").json == []


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
# Room-scoped access -- a caregiver should only see cameras/incidents/clips
# for rooms they're explicitly assigned to; admins see everything.
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
    # A caregiver created with no assigned_rooms (the actual default) --
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
    assert r.status_code == 404  # not 403 -- must not confirm the incident exists
    assert cg.post(f"/api/incidents/{other_room_incident['id']}/notes", json={"notes": "x"}).status_code == 404


def test_caregiver_cannot_reach_video_feed_or_score_in_other_room(app_client):
    _seed_two_room_setup(app_client)
    cg = login(app_client, "jane@ward.org")
    assert cg.get("/video_feed/CAM-02").status_code == 404
    assert cg.get("/api/score/CAM-02").status_code == 404
    # Her own assigned camera's routes should NOT 404 for the same reason.
    assert cg.get("/api/score/CAM-01").status_code == 200


def test_deleted_camera_video_feed_and_score_deny_not_allow(app_client):
    """Regression test for a bug found in the final code sweep: _camera_room()
    returns None for a camera id that isn't in cameras.json (e.g. deleted
    while its worker thread was still registered), and video_feed/get_score
    used to treat "room is None" as "no room check applies" -- the opposite
    of what _clip_room_accessible does for the same ambiguous case. That
    let a caregiver reach a deleted camera's still-running feed/score even
    though they could never have accessed it by camera id while it existed
    unassigned to their rooms. Deleting must narrow access, never widen it."""
    _seed_two_room_setup(app_client)
    admin = login(app_client, "lead@ward.org")
    # A worker some request handler still has a reference to, as if the
    # camera thread never actually stopped -- see delete_camera's comment.
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
