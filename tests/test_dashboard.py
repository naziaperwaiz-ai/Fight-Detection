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

    users_module.create_caregiver("jane@ward.org", "correcthorse123", "Jane Doe", role="caregiver")
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
    # Camera add/edit/delete is admin-only (see app.py) -- a caregiver
    # account can still read camera status, just not change what exists.
    admin = login(app_client, "lead@ward.org")
    assert admin.post("/api/cameras/add", json={
        "id": "CAM-01", "room": "Sunroom Wing", "source": "0", "threshold": 0.9, "patients": 3, "priority": "High",
    }).status_code == 200

    cams = admin.get("/api/cameras").json
    assert cams[0]["patients"] == 3 and cams[0]["priority"] == "High"

    assert admin.post("/api/cameras/update/CAM-01", json={"room": "Renamed"}).status_code == 200
    assert admin.delete("/api/cameras/delete/CAM-01").status_code == 200
    assert admin.get("/api/cameras").json == []


def test_caregiver_can_view_but_not_manage_cameras(app_client):
    admin = login(app_client, "lead@ward.org")
    admin.post("/api/cameras/add", json={"id": "CAM-01", "room": "Sunroom Wing", "source": "0"})

    cg = login(app_client, "jane@ward.org")
    assert cg.get("/api/cameras").status_code == 200  # read is open to any caregiver

    assert cg.post("/api/cameras/add", json={"id": "CAM-02", "room": "Rehab Wing", "source": "1"}).status_code == 403
    assert cg.post("/api/cameras/update/CAM-01", json={"room": "Hacked"}).status_code == 403
    assert cg.delete("/api/cameras/delete/CAM-01").status_code == 403

    # confirm none of the blocked calls actually changed anything
    cams = admin.get("/api/cameras").json
    assert len(cams) == 1 and cams[0]["room"] == "Sunroom Wing"


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
