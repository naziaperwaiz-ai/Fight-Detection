# src/dashboard/app.py
from flask import (
    Flask, render_template, jsonify, request, send_from_directory,
    Response, redirect, url_for
)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import json, os, cv2, time, secrets, hmac, uuid, re
import threading

from dashboard.retention import cleanup_events, cleanup_clips
from dashboard.events_store import SqliteEventsStore, SqliteCheckinsStore

from auth.users import (
    verify_caregiver, get_caregiver_by_id,
    create_invite, get_valid_invite, consume_invite,
    list_pending_invites, revoke_invite, revoke_invite_by_email,
    list_caregivers, set_assigned_rooms,
)

app = Flask(__name__)

_BASE = Path(__file__).parent.parent.parent
CLIPS_DIR       = _BASE / "outputs" / "clips"
MODELS_DIR      = _BASE / "models"
LOGS_FILE       = _BASE / "outputs" / "logs" / "events.json"
EVENTS_DB_FILE  = _BASE / "outputs" / "logs" / "events.db"
CAMERAS_FILE    = _BASE / "outputs" / "logs" / "cameras.json"
PROFILES_FILE   = _BASE / "outputs" / "logs" / "profiles.json"
ANNOUNCE_FILE   = _BASE / "outputs" / "logs" / "announcements.json"
SETTINGS_FILE   = _BASE / "outputs" / "logs" / "alert_settings.json"
SYS_SETTINGS_FILE = _BASE / "outputs" / "logs" / "system_settings.json"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Secret key
# ---------------------------------------------------------------------------
try:
    from detection.config import Config
    _cfg = Config()
except Exception:
    _cfg = None

app.secret_key = (getattr(_cfg, "SECRET_KEY", None) if _cfg else None) or secrets.token_hex(32)
_INTERNAL_API_KEY = getattr(_cfg, "INTERNAL_API_KEY", None) if _cfg else None

# ---------------------------------------------------------------------------
# Session cookie hardening
# ---------------------------------------------------------------------------
# SESSION_COOKIE_HTTPONLY is already Flask's default (True); set explicitly
# so this block is a complete, self-contained statement of the intended
# cookie policy rather than half-implied by framework defaults.
# SESSION_COOKIE_SECURE=True means the browser will never send the session
# cookie over a plain HTTP connection, even if one is briefly reachable (a
# misconfigured proxy, someone hitting the LAN IP on the HTTP port
# directly). main.py's _ssl_context() falls back to plain HTTP when no TLS
# cert is present so a fresh checkout still runs, but that fallback should
# never be able to leak a session cookie in cleartext as a side effect;
# generate a cert (see src/certs/generate_cert.py) for any real deployment.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# ---------------------------------------------------------------------------
# CSRF protection
# ---------------------------------------------------------------------------
# Every state-changing route in this file is a plain session-cookie-
# authenticated POST/DELETE with no CSRF token, which is exploitable for
# any route that accepts a "simple" request content type (multipart/
# form-data, form-urlencoded) that a cross-site <form> can submit without
# a CORS preflight -- see upload_model()'s file-write endpoint below, which
# is exactly that. The JSON fetch() calls from dashboard.js are already
# incidentally protected by the CORS preflight browsers require for a
# non-simple Content-Type, since _strip_wildcard_cors below never grants a
# cross-origin an Allow-Origin -- but relying on that as the only CSRF
# defense is fragile (it depends on every future route staying JSON-only)
# and does nothing for the multipart upload route. CSRFProtect covers
# every mutating route uniformly; dashboard.js's fetch() wrapper attaches
# the token as a header automatically (see the top of dashboard.js), and
# login.html/signup.html carry it as a hidden form field.
csrf = CSRFProtect(app)

# ---------------------------------------------------------------------------
# CORS hardening
# ---------------------------------------------------------------------------
# Same-origin only. No CORS package is used anywhere in this project, and no
# route should ever send Access-Control-Allow-Origin: *. This hook is a
# defense-in-depth backstop against a future change reintroducing one.
@app.after_request
def _strip_wildcard_cors(resp):
    if resp.headers.get("Access-Control-Allow-Origin") == "*":
        resp.headers.pop("Access-Control-Allow-Origin", None)
        resp.headers.pop("Access-Control-Allow-Credentials", None)
    return resp

# ---------------------------------------------------------------------------
# Security response headers
# ---------------------------------------------------------------------------
# script-src still needs 'unsafe-inline': index.html's templates use
# static onclick="Haven.method(...)" handlers throughout (with no
# interpolated data -- e.g. onclick="Haven.goPage('cameras')" -- so they
# were never part of the XSS finding), and a strict CSP blocks every
# inline event handler regardless of what it contains. Rewriting all of
# those to addEventListener wiring is a real improvement worth doing, but
# it's a template-wide refactor, not part of fixing the reported
# vulnerabilities, so it's left as follow-up rather than done here as a
# side effect. The XSS this CSP would otherwise be the last line of
# defense for (finding 2.2, admin-controlled data landing in an onclick
# attribute) is already closed at the source by dashboard.js's
# data-action delegated-click handler, which stopped interpolating that
# data into any onclick attribute at all -- so 'unsafe-inline' here does
# not reopen that hole. frame-ancestors and nosniff below are unaffected
# by this and are fully enforced.
@app.after_request
def _security_headers(resp):
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "frame-ancestors 'none'"
    )
    return resp

# ---------------------------------------------------------------------------
# Auth setup
# ---------------------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# In-memory storage is fine for a single-process deployment (the default
# here). If this ever runs behind multiple worker processes/instances,
# point storage_uri at a shared Redis instance instead. In-memory limits
# are per-process and are not shared across workers, which would let an
# attacker get more attempts than intended by hitting a different worker
# each time.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)


class Caregiver(UserMixin):
    def __init__(self, record):
        self.id    = record["id"]
        self.email = record["email"]
        self.name  = record.get("name", record["email"])
        self.role  = record.get("role", "caregiver")
        self.assigned_rooms = record.get("assigned_rooms", [])

    @property
    def is_admin(self):
        return self.role == "admin"

    def can_access_room(self, room):
        # Admins see every room. Room scoping exists to keep a caregiver
        # from seeing patients they have no care relationship with, not
        # to limit facility-wide oversight. A missing or empty
        # assigned_rooms means no rooms are accessible; see
        # auth/users.py's create_caregiver for why that default is
        # deliberate.
        return self.is_admin or room in self.assigned_rooms


@login_manager.user_loader
def load_user(user_id):
    record = get_caregiver_by_id(user_id)
    return Caregiver(record) if record else None


@login_manager.unauthorized_handler
def _unauthorized():
    if request.path.startswith(("/api/", "/video_feed/", "/clips/")):
        return jsonify({"error": "authentication required"}), 401
    return redirect(url_for("login", next=request.path))


def admin_required(fn):
    """Gate model/detection-default edits to admin accounts. Caregivers get
    a clean 403. The frontend also hides these controls, but the backend
    check is what actually enforces it."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({"error": "administrator access required"}), 403
        return fn(*args, **kwargs)
    return wrapper


def internal_key_required(fn):
    """Guards service-to-service endpoints (called by CameraWorker, not by
    a caregiver's browser)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _INTERNAL_API_KEY:
            return jsonify({"error": "internal API key not configured"}), 503
        supplied = request.headers.get("X-Internal-Key", "")
        if not hmac.compare_digest(supplied, _INTERNAL_API_KEY):
            return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# JSON stores
#
# Every store below is a JsonStore: a per-file threading.Lock plus an
# atomic (temp-file + os.replace) write, instead of the old bare
# read-whole-file / write-whole-file functions. Two problems with the old
# version, both real under concurrent requests (two caregivers editing at
# once, or a caregiver's request landing while CameraWorker's add_event
# posts, or while the daily retention thread runs):
#
#   1. Lost updates: load() then save() is a read-modify-write with no
#      lock across the gap. Two requests that both load the same old
#      version race to save; whichever finishes last wins, and the
#      other's change is silently discarded rather than merged or
#      erroring. mutate() (and mutate_if()) close this by holding the
#      lock for the entire load-modify-save cycle, not just around each
#      individual read or write call.
#   2. Torn writes: path.write_text() truncates the file before writing
#      the new content. A reader (or a process crash) mid-write could
#      see an empty file, or -- worse -- a JSON parse error on a file
#      that is neither the old nor the new valid content. _write_json_
#      atomic() writes to a temp file in the same directory first, then
#      os.replace()s it into place; os.replace is atomic on POSIX and
#      Windows (since Python 3.3), so any reader always sees either the
#      fully-old or fully-new file, never a partial one.
# ---------------------------------------------------------------------------
_workers = {}  # cam_id -> CameraWorker

def register_worker(cam_id, worker):
    _workers[cam_id] = worker

def _read_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return default
    return default

_MISSING = object()   # sentinel: "the file does not exist / could not be parsed", distinct from any real stored value

def _write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(path))
    finally:
        # Only ever reached with tmp still on disk if write_text() or
        # os.replace() raised before/without completing the rename;
        # os.replace() removes/renames the temp file away on success, so
        # this is a no-op on the normal path, not a redundant delete.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class JsonStore:
    """One JSON file, one lock. See the module docstring above this class
    for why plain load()/save() functions were not race-safe."""

    def __init__(self, path, default_factory):
        self.path = path
        self.default_factory = default_factory
        self._lock = threading.Lock()

    def _load_locked(self):
        data = _read_json(self.path, _MISSING)
        return self.default_factory() if data is _MISSING else data

    def _save_locked(self, data):
        _write_json_atomic(self.path, data)

    def load(self):
        with self._lock:
            return self._load_locked()

    def save(self, data):
        with self._lock:
            self._save_locked(data)

    def mutate(self, fn):
        """Hold the lock across the entire read-modify-write cycle. `fn`
        receives the freshly-loaded data and is expected to mutate it in
        place (list.append, dict item assignment, etc.); the (possibly
        mutated) data is always persisted afterward. Returns whatever
        `fn` returns, unchanged, so a caller can pass outcome
        information (found/authorized/computed fields) back without a
        second, separately-locked round trip to the store."""
        with self._lock:
            data = self._load_locked()
            result = fn(data)
            self._save_locked(data)
            return result

    def mutate_if(self, fn):
        """Like mutate(), but `fn` returns a bool: True to persist the
        (mutated) data, False to skip the write entirely. Use this
        instead of mutate() when the mutation is often a no-op and an
        unconditional write would be wasted disk I/O -- see run_cleanup,
        which runs every 24h against potentially thousands of events but
        usually deletes nothing."""
        with self._lock:
            data = self._load_locked()
            if fn(data):
                self._save_locked(data)
            return data

    def load_or_seed(self, seed_factory):
        """Like load(), but if the file does not exist at all yet (not
        merely an empty list/dict that was legitimately saved that way),
        atomically writes and returns seed_factory()'s result instead of
        the plain default. See load_announcements below."""
        with self._lock:
            existing = _read_json(self.path, _MISSING)
            if existing is not _MISSING:
                return existing
            seeded = seed_factory()
            self._save_locked(seeded)
            return seeded


_events_store        = SqliteEventsStore(EVENTS_DB_FILE, legacy_json_path=LOGS_FILE)
_cameras_store       = JsonStore(CAMERAS_FILE, list)
_profiles_store      = JsonStore(PROFILES_FILE, dict)
_announce_store      = JsonStore(ANNOUNCE_FILE, list)
_sys_settings_store  = JsonStore(SYS_SETTINGS_FILE, dict)
# Second table in the same events.db file, not a separate JSON store --
# see events_store.py's SqliteCheckinsStore docstring for why.
_checkins_store      = SqliteCheckinsStore(EVENTS_DB_FILE)


def _default_alert_settings():
    return {
        "recipients": [], "threshold": 90, "cooldown": 120, "email_channel": True,
        # Client-side chime in the dashboard tab when a new alert arrives.
        # Purely a browser-side setting (no server-side sound to send), but
        # stored here so it persists/syncs the same way email_channel does.
        "sound_channel": True,
        # OS-level desktop notification (Notification API) while the tab
        # is open. Also purely client-side; the server never sends
        # anything for this channel, it just remembers the caregiver's
        # preference. Requires a secure context (HTTPS or localhost) in
        # the browser; see the warning dashboard.js surfaces otherwise.
        "desktop_channel": True,
    }


_alert_settings_store = JsonStore(SETTINGS_FILE, _default_alert_settings)


def load_events():
    return _events_store.load()

def save_events(events):
    _events_store.save(events)

def load_cameras():
    # Defaults to an empty list rather than a canned "CAM-01" seed. A
    # seed camera here would persist once a real camera gets added on
    # top of it (add_camera loads then appends), leaving a fake camera
    # in the list forever. A fresh install should show zero cameras
    # until one is added.
    return _cameras_store.load()

def save_cameras(cameras):
    _cameras_store.save(cameras)

def _camera_room(cam_id):
    """Room name for a camera id, or None if the camera doesn't exist.
    Used to gate access to a specific camera's feed/score/clips by room,
    since those routes only get a camera id, not a room, from the URL."""
    for c in load_cameras():
        if c["id"] == cam_id:
            return c.get("room")
    return None

def load_profiles():
    return _profiles_store.load()

def save_profiles(profiles):
    _profiles_store.save(profiles)

def load_announcements():
    # Seed with example shift notes on first run so the panel isn't
    # empty. load_or_seed only seeds when the file has never existed, so
    # this cannot repeatedly reseed on every read.
    return _announce_store.load_or_seed(lambda: [{
        "time": datetime.now().strftime("%H:%M"),
        "text": "Shift handover complete.",
        "icon": "icon-shift-handover.jpg",
        "author": "system",
    }])

def save_announcements(items):
    _announce_store.save(items)

def load_settings():
    return _alert_settings_store.load()

def save_settings(s):
    _alert_settings_store.save(s)

def _system_settings_with_defaults(stored):
    """Pure merge: Config-derived defaults, overridden by whatever is
    actually stored in system_settings.json. Factored out from
    load_system_settings so update_system_settings/upload_model can
    compute the same effective settings from inside a locked mutate()
    cycle (reusing the raw `stored` dict mutate() already loaded) instead
    of calling back into load_system_settings and re-reading the file a
    second time, unlocked, mid-mutation."""
    defaults = {
        "confirm_seconds": 3, "motion_threshold": 1.5, "buffer_seconds": 10, "post_event_seconds": 15,
        # Retention: how long an incident record or clip file is kept
        # before automatic deletion. false_positive_retention_days is
        # shorter on purpose, since confirmed noise does not need the
        # full window. See dashboard/retention.py.
        "retention_days": 90,
        "false_positive_retention_days": 7,
    }
    if _cfg is not None:
        defaults["confirm_seconds"]    = getattr(_cfg, "CONFIRM_SECONDS", defaults["confirm_seconds"])
        defaults["motion_threshold"]   = getattr(_cfg, "MOTION_THRESHOLD", defaults["motion_threshold"])
        defaults["buffer_seconds"]     = getattr(_cfg, "BUFFER_SECONDS", defaults["buffer_seconds"])
        defaults["post_event_seconds"] = getattr(_cfg, "POST_EVENT_SECONDS", defaults["post_event_seconds"])
    if stored:
        defaults.update(stored)
    return defaults

def load_system_settings():
    return _system_settings_with_defaults(_sys_settings_store.load())

def save_system_settings(s):
    _sys_settings_store.save(s)


def generate_frames(cam_id):
    while True:
        worker = _workers.get(cam_id)
        if worker:
            frame = worker.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        email    = request.form.get("email", "")
        password = request.form.get("password", "")
        record   = verify_caregiver(email, password)

        if record:
            login_user(Caregiver(record), remember=bool(request.form.get("remember")))
            next_url = request.args.get("next") or request.form.get("next")
            if next_url and next_url.startswith("/") and not next_url.startswith("//"):
                return redirect(next_url)
            return redirect(url_for("index"))

        error = "Invalid email or password."

    return render_template("login.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Sign-up (invite-only)
#
# There is intentionally no route that lets a visitor create an account
# by just showing up here; that would undo the access control the
# caregiver login was built for. The only way to reach a working sign-up
# form is to hold a token an admin generated via POST /api/invites (see
# below). The
# token is long, random, single-use, and expires; get_valid_invite() checks
# all three before this route renders anything.
# ---------------------------------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    token  = request.args.get("token") or request.form.get("token", "")
    invite = get_valid_invite(token)
    if not invite:
        return render_template("signup.html", invite=None, error=None)

    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        if password != confirm:
            error = "Passwords do not match."
        else:
            try:
                record = consume_invite(token, password)
            except ValueError as e:
                error = str(e)
            else:
                login_user(Caregiver(record))
                return redirect(url_for("index"))

    return render_template("signup.html", invite=invite, error=error, token=token)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html", caregiver_name=current_user.name, is_admin=current_user.is_admin)

@app.route("/video_feed/<cam_id>")
@login_required
def video_feed(cam_id):
    room = _camera_room(cam_id)
    # Deny, don't allow, when the camera id isn't in cameras.json (unknown,
    # or deleted while its worker thread was still running/registered --
    # see delete_camera). Room being None must never mean "no room check
    # applies"; it must mean "can't prove access, so refuse." Matches
    # _clip_room_accessible's default-deny for the same ambiguous case.
    if room is None or not current_user.can_access_room(room):
        return jsonify({"error": "not found"}), 404
    return Response(generate_frames(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/score/<cam_id>")
@login_required
def get_score(cam_id):
    room = _camera_room(cam_id)
    if room is None or not current_user.can_access_room(room):
        return jsonify({"error": "not found"}), 404
    worker = _workers.get(cam_id)
    return jsonify({"score": worker.score if worker else 0.0})

@app.route("/api/me")
@login_required
def get_me():
    return jsonify({"id": current_user.id, "email": current_user.email, "name": current_user.name, "role": current_user.role})


# ---------------------------------------------------------------------------
# Events / incidents
# ---------------------------------------------------------------------------
def _visible_events(events):
    """Filter events to rooms current_user can access. Admins see all."""
    if current_user.is_admin:
        return events
    return [e for e in events if e.get("room") in current_user.assigned_rooms]

@app.route("/api/events")
@login_required
def get_events():
    return jsonify(_visible_events(load_events())[-200:])

@app.route("/api/events/add", methods=["POST"])
@csrf.exempt   # server-to-server (CameraWorker), authenticated by X-Internal-Key, no browser session/cookie involved
@internal_key_required
@limiter.limit("60 per minute")
def add_event():
    # Called by CameraWorker (server-to-server) via the internal API key,
    # not by a caregiver's browser session.
    data = request.json or {}
    def _mutate(events):
        events.append({
            "id":         uuid.uuid4().hex[:8].upper(),
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id":  data.get("camera_id", "unknown"),
            "room":       data.get("room", "unknown"),
            "event_type": data.get("event_type", "unknown"),
            "confidence": data.get("confidence", 0.0),
            "clip_path":  data.get("clip_path", ""),
            "states":     data.get("states", []),  # snapshot of tracked-person states at alert time
            "detail":     data.get("detail", ""),  # free-text rule explanation (e.g. hazard events)
            "notes":      "",
            "reviewed":   False,
            "false_positive": False,
        })
    _events_store.mutate(_mutate)
    return jsonify({"status": "ok"})

@app.route("/api/incidents/<incident_id>")
@login_required
def get_incident(incident_id):
    for e in load_events():
        if e.get("id") == incident_id:
            if not current_user.can_access_room(e.get("room")):
                return jsonify({"error": "not found"}), 404
            return jsonify(e)
    return jsonify({"error": "not found"}), 404

def _mutate_incident(incident_id, apply_fn):
    """Shared by set_incident_notes/toggle_incident_review/
    toggle_incident_false_positive: looks up incident_id from inside the
    same locked mutate() cycle that will persist any change, instead of
    an earlier, separately-locked load(). Two requests editing the same
    incident (or one editing while a retention cleanup pass deletes it)
    can no longer have the later save silently overwrite the earlier
    one's change, since both now serialize on _events_store's lock for
    the whole find-and-mutate step, not just the final write.

    apply_fn(e) mutates the matched event dict in place and returns
    either None (respond with just {"status": "ok"}) or a dict of extra
    fields to merge into that response (e.g. {"reviewed": True}).

    Returns {"found": bool, "authorized": bool, "extra": dict|None}.
    """
    outcome = {"found": False, "authorized": True, "extra": None}
    def _mutate(events):
        for e in events:
            if e.get("id") == incident_id:
                outcome["found"] = True
                if not current_user.can_access_room(e.get("room")):
                    outcome["authorized"] = False
                    return
                outcome["extra"] = apply_fn(e)
                return
    _events_store.mutate(_mutate)
    return outcome

@app.route("/api/incidents/<incident_id>/notes", methods=["POST"])
@login_required
def set_incident_notes(incident_id):
    data = request.json or {}
    def _apply(e):
        e["notes"] = data.get("notes", "")
        return None
    outcome = _mutate_incident(incident_id, _apply)
    if not outcome["found"] or not outcome["authorized"]:
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": "ok"})

@app.route("/api/incidents/<incident_id>/review", methods=["POST"])
@login_required
def toggle_incident_review(incident_id):
    def _apply(e):
        e["reviewed"] = not e.get("reviewed", False)
        return {"reviewed": e["reviewed"]}
    outcome = _mutate_incident(incident_id, _apply)
    if not outcome["found"] or not outcome["authorized"]:
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": "ok", **outcome["extra"]})

@app.route("/api/incidents/<incident_id>/false-positive", methods=["POST"])
@login_required
def toggle_incident_false_positive(incident_id):
    def _apply(e):
        e["false_positive"] = not e.get("false_positive", False)
        return {"false_positive": e["false_positive"]}
    outcome = _mutate_incident(incident_id, _apply)
    if not outcome["found"] or not outcome["authorized"]:
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": "ok", **outcome["extra"]})


# ---------------------------------------------------------------------------
# Cameras: any caregiver can view status and the live preview, but adding,
# editing, or deleting a camera is admin-only. Which physical cameras
# exist is a facility-configuration decision, not a day-to-day
# caregiving one. A caregiver account being able to silently remove
# coverage from a room, accidentally or otherwise, with no confirmation
# and no audit trail was an oversight from before the caregiver/admin
# role split existed, not a deliberate choice, and is closed here.
# ---------------------------------------------------------------------------
@app.route("/api/cameras")
@login_required
def get_cameras():
    cameras = load_cameras()
    if not current_user.is_admin:
        cameras = [c for c in cameras if current_user.can_access_room(c.get("room"))]
    for c in cameras:
        c.setdefault("patients", 0)
        c.setdefault("priority", "Medium")
        # A worker thread existing for this camera is our real liveness
        # signal (it means the process actually attempted to open the
        # source), combined with the user-set active flag.
        c["liveStatus"] = "Live" if (c.get("active") and c["id"] in _workers) else "Offline"
    return jsonify(cameras)

@app.route("/api/cameras/add", methods=["POST"])
@admin_required
def add_camera():
    data = request.json or {}
    # Falls back to the facility-wide Alert Settings threshold (0-100,
    # percent) rather than a hardcoded 0.7, so a client that omits
    # threshold (or an older client built before the per-camera slider
    # existed) still gets the sensitivity an admin already configured
    # globally, not an unrelated hardcoded default.
    default_threshold = load_settings().get("threshold", 90) / 100.0

    def _mutate(cameras):
        # The id-default (CAM-<count+1>) is computed here, inside the
        # locked mutate cycle, against the list mutate() just loaded --
        # not against a copy fetched before the lock was taken. Two
        # concurrent add_camera calls that both read "3 existing
        # cameras" before either saved used to both default to
        # "CAM-04", silently colliding; serializing the read and the
        # id-assignment together closes that.
        cameras.append({
            "id":        data.get("id", f"CAM-{len(cameras)+1:02d}"),
            "room":      data.get("room", "Unknown Room"),
            "source":    data.get("source", "0"),
            "threshold": float(data.get("threshold", default_threshold)),
            # Honor a client-supplied active flag (the camera modal's "Active"
            # checkbox already sends one, see dashboard.js's saveCamera)
            # instead of forcing every newly added camera to active
            # regardless of what was submitted. Still defaults to True when
            # omitted, matching the previous behavior for any caller that
            # does not send this field.
            "active":    bool(data.get("active", True)),
            "patients":  int(data.get("patients", 0)),
            "priority":  data.get("priority", "Medium"),
            # Per-camera opt-in for the dangerous-object-near-wrist rule (see
            # detection/hazard.py). Defaults to off: hazard detection adds a
            # second model (pose) to every batched cycle, and a facility
            # should choose it deliberately per room rather than have it
            # silently on. See main.py's _build_camera_config for how this
            # becomes each camera's HAZARD_DETECTION_ENABLED, and
            # multi_camera.py's module docstring for which hazard settings
            # are necessarily shared across cameras (severity, pose weights,
            # image size) versus genuinely per-camera (this flag, proximity,
            # debounce length).
            "hazard_enabled": bool(data.get("hazard_enabled", False)),
        })
    _cameras_store.mutate(_mutate)
    return jsonify({"status": "ok"})

@app.route("/api/cameras/delete/<cam_id>", methods=["DELETE"])
@admin_required
def delete_camera(cam_id):
    def _mutate(cameras):
        cameras[:] = [c for c in cameras if c["id"] != cam_id]
    _cameras_store.mutate(_mutate)
    # Unregister the running worker, if any, so /video_feed and
    # /api/score for this id stop being servable the instant the camera
    # is deleted. Without this, a worker thread started at process
    # launch keeps capturing and streaming after its camera record is
    # gone, and the dashboard has no route left that will show it as
    # "Offline" to warn anyone. This does not stop the worker's
    # background thread itself; CameraWorker has no stop/join mechanism
    # yet, tracked as follow-up work. It does close off every access
    # path to what that thread is still capturing.
    _workers.pop(cam_id, None)
    return jsonify({"status": "ok"})

@app.route("/api/cameras/update/<cam_id>", methods=["POST"])
@admin_required
def update_camera(cam_id):
    data = request.json or {}
    def _mutate(cameras):
        for c in cameras:
            if c["id"] == cam_id:
                c.update(data)
    _cameras_store.mutate(_mutate)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
_CLIP_NAME_RE = re.compile(r"^alert_(?P<cam>.+)_(?P<ts>\d{8}_\d{6})\.mp4$")

def _clip_room_accessible(filename):
    """A clip's filename encodes its camera id, not its room, so this maps
    camera_id -> room via the current camera list. If the camera was
    since deleted (an orphaned clip) or the filename does not match the
    expected pattern, this defaults to admin-only: a caregiver's
    assignment cannot be verified against a room that cannot be
    determined, and defaulting to visible for anything ambiguous would
    defeat the point of this check.
    """
    if current_user.is_admin:
        return True
    m = _CLIP_NAME_RE.match(filename)
    if not m:
        return False
    cam_id = m.group("cam")
    room = _camera_room(cam_id)
    if room is None:
        return False
    return current_user.can_access_room(room)

@app.route("/clips/<filename>")
@login_required
def serve_clip(filename):
    # Reject anything that isn't a real clip filename before it reaches
    # the filesystem or the room-access check, admin or not. Without
    # this, an admin session hits send_from_directory for a name that
    # can never exist -- most commonly "Saving..." (see
    # CLIP_SAVING_PLACEHOLDER on the dashboard: the placeholder clip_path
    # an incident carries between "alert fired" and "clip finished
    # encoding") -- and gets whatever generic 404 Werkzeug happens to
    # render for a missing file, instead of the same clear JSON error a
    # non-admin gets from _clip_room_accessible below for the same
    # not-a-real-clip filename.
    if not _CLIP_NAME_RE.match(filename):
        return jsonify({"error": "not found"}), 404
    if not _clip_room_accessible(filename):
        return jsonify({"error": "not found"}), 404
    # mimetype is passed explicitly rather than left for Werkzeug to guess
    # from the extension. Flask/Werkzeug's guess goes through Python's
    # stdlib mimetypes module, which on Windows consults the Windows
    # registry's file-extension associations rather than a bundled table.
    # When that registry mapping for .mp4 is missing or wrong (a real,
    # fairly common Windows misconfiguration -- unrelated to this
    # project, and easy to hit if a codec pack or "cleaner" tool ever
    # touched HKEY_CLASSES_ROOT), the guess silently falls through to
    # application/octet-stream. The clip on disk is a perfectly valid,
    # fully playable H.264 file at that point -- confirmed with ffprobe
    # -- but the browser's <video> element never attempts to decode a
    # response it wasn't told is a video, and shows the same dead
    # 0:00/black-frame state a genuinely broken file would. Every clip
    # this route ever serves is a .mp4 by construction (_CLIP_NAME_RE
    # above already enforces that), so there is nothing to actually
    # guess -- hardcoding it removes the host OS as a variable entirely.
    return send_from_directory(
        str(CLIPS_DIR.absolute()), filename, mimetype="video/mp4"
    )

@app.route("/api/clips")
@login_required
def get_clips():
    clips = sorted(CLIPS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for c in clips:
        if not _clip_room_accessible(c.name):
            continue
        m = _CLIP_NAME_RE.match(c.name)
        out.append({
            "filename": c.name,
            "camera_id": m.group("cam") if m else "unknown",
            "timestamp": m.group("ts") if m else "",
            "mtime": c.stat().st_mtime,
        })
        if len(out) >= 100:
            break
    return jsonify(out)


# ---------------------------------------------------------------------------
# Per-caregiver profile (row-level scoping: always current_user.id,
# never a client-supplied id; see the note on /api/me in the design doc).
# ---------------------------------------------------------------------------
@app.route("/api/profile")
@login_required
def get_profile():
    profiles = load_profiles()
    p = profiles.get(current_user.id, {"about": "", "notes": ""})
    p["name"] = current_user.name
    p["role"] = current_user.role
    return jsonify(p)

@app.route("/api/profile", methods=["POST"])
@login_required
def update_profile():
    data = request.json or {}
    def _mutate(profiles):
        profiles[current_user.id] = {
            "about": data.get("about", ""),
            "notes": data.get("notes", ""),
        }
    _profiles_store.mutate(_mutate)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Announcements (shift notes)
# ---------------------------------------------------------------------------
@app.route("/api/announcements")
@login_required
def get_announcements():
    return jsonify(load_announcements()[-20:])

@app.route("/api/announcements", methods=["POST"])
@login_required
def add_announcement():
    data = request.json or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    def _mutate(items):
        # Seed-on-first-run, same note as load_announcements, in case
        # this is the very first write this deployment ever makes (a
        # POST with no prior GET). load_or_seed isn't used here since
        # this call is already inside _announce_store's own lock via
        # mutate(); nesting a second load_or_seed() call would deadlock
        # on threading.Lock (non-reentrant).
        if not items:
            items.append({
                "time": datetime.now().strftime("%H:%M"),
                "text": "Shift handover complete.",
                "icon": "icon-shift-handover.jpg",
                "author": "system",
            })
        items.append({
            "time": datetime.now().strftime("%H:%M"),
            "text": text[:500],
            "author": current_user.name,
            "icon": None,
        })
    _announce_store.mutate(_mutate)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Round check-ins ("I've walked through my assigned zones"). Logging and
# reading your own last check-in are row-scoped by session identity like
# /api/profile above: always current_user, never a client-supplied
# caregiver id, so one caregiver's dashboard can't be used to log or
# read a check-in as someone else. Stored in a second table in
# events.db (SqliteCheckinsStore) rather than a JSON file -- see that
# class's docstring in events_store.py for why this data moved there
# even though its write volume doesn't scale with camera count the way
# incident events' does.
# ---------------------------------------------------------------------------
@app.route("/api/checkins/add", methods=["POST"])
@login_required
def add_checkin():
    record = {
        "id":            uuid.uuid4().hex[:8].upper(),
        "caregiver_id":  current_user.id,
        "caregiver_name": current_user.name,
        # Snapshotted at check-in time rather than looked up later, so
        # the record still reflects what this caregiver was actually
        # covering even if their room assignment changes afterward.
        "rooms":         "all" if current_user.is_admin else list(current_user.assigned_rooms),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _checkins_store.add(record)
    return jsonify(record)

@app.route("/api/checkins/last")
@login_required
def get_last_checkin():
    return jsonify(_checkins_store.last_for(current_user.id))

@app.route("/api/checkins")
@admin_required
def get_checkin_history():
    # Admin-only audit view across every caregiver, not just the
    # caller's own check-ins -- see /api/checkins/last above for the
    # self-scoped version any caregiver can read. Rounds accountability
    # (who checked in, when, and whether anyone is skipping them) is
    # exactly the kind of thing that should be visible to a shift lead,
    # not just to the person doing the check-in.
    return jsonify(_checkins_store.list_all())


# ---------------------------------------------------------------------------
# Alert settings (any caregiver can view/edit, matching how the
# physical on-call sheet already works: whoever's on shift can update it)
# ---------------------------------------------------------------------------
@app.route("/api/settings")
@login_required
def get_settings():
    return jsonify(load_settings())

@app.route("/api/settings", methods=["POST"])
@login_required
def update_settings():
    data = request.json or {}
    def _mutate(s):
        if "recipients" in data:
            s["recipients"] = [r for r in data["recipients"] if isinstance(r, str) and "@" in r]
        if "threshold" in data:
            s["threshold"] = max(50, min(100, int(data["threshold"])))
        if "cooldown" in data:
            s["cooldown"] = max(30, min(300, int(data["cooldown"])))
        if "email_channel" in data:
            s["email_channel"] = bool(data["email_channel"])
        if "sound_channel" in data:
            s["sound_channel"] = bool(data["sound_channel"])
        if "desktop_channel" in data:
            s["desktop_channel"] = bool(data["desktop_channel"])
    _alert_settings_store.mutate(_mutate)
    return jsonify({"status": "ok"})

@app.route("/api/settings/test-alert", methods=["POST"])
@login_required
@limiter.limit("5 per minute")
def test_alert():
    s = load_settings()
    recipients = s.get("recipients", [])
    if not recipients:
        return jsonify({"error": "No recipients configured. Add one first."}), 400
    if not s.get("email_channel", True):
        return jsonify({"error": "Email channel is turned off."}), 400
    try:
        from notification.notifier import Notifier
        cfg = Config() if _cfg is not None else None
        if cfg is None:
            return jsonify({"error": "Email is not configured (missing config.py)."}), 503
        cfg.EMAIL_RECIPIENTS = recipients
        Notifier(cfg).send_alert("TEST", "N/A", "Test Alert", 0.0, "")
    except Exception as e:
        return jsonify({"error": f"Could not send test alert: {e}"}), 500
    return jsonify({"status": "ok", "recipients": recipients})


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
@app.route("/api/analytics")
@login_required
def get_analytics():
    days = max(1, min(90, int(request.args.get("days", 7))))
    events = _visible_events(load_events())
    cutoff = datetime.now() - timedelta(days=days)

    def parsed(e):
        try:
            return datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            return None

    in_range = [(e, parsed(e)) for e in events]
    in_range = [(e, t) for e, t in in_range if t and t >= cutoff]

    total = len(in_range)
    avg_conf = round(sum(e.get("confidence", 0) for e, _ in in_range) / total * 100, 1) if total else 0.0

    by_room = {}
    by_camera = {}
    by_day = {}
    for e, t in in_range:
        by_room[e.get("room", "unknown")] = by_room.get(e.get("room", "unknown"), 0) + 1
        by_camera[e.get("camera_id", "unknown")] = by_camera.get(e.get("camera_id", "unknown"), 0) + 1
        day_key = t.strftime("%Y-%m-%d")
        by_day[day_key] = by_day.get(day_key, 0) + 1

    busiest_room = max(by_room, key=by_room.get) if by_room else "—"
    busiest_camera = max(by_camera, key=by_camera.get) if by_camera else "—"

    day_series = []
    for i in range(days - 1, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        label = (datetime.now() - timedelta(days=i)).strftime("%a")
        day_series.append({"date": d, "label": label, "count": by_day.get(d, 0)})

    room_series = sorted(by_room.items(), key=lambda kv: kv[1], reverse=True)[:8]

    return jsonify({
        "total_incidents": total,
        "avg_confidence_pct": avg_conf,
        "busiest_room": busiest_room,
        "busiest_camera": busiest_camera,
        "by_day": day_series,
        "by_room": [{"room": r, "count": c} for r, c in room_series],
    })


# ---------------------------------------------------------------------------
# Retention: age-based cleanup of incident records and clip files. See
# dashboard/retention.py for the actual deletion logic; this is just the
# wiring (settings-driven windows, a manual admin trigger, and a daily
# background pass so this does not rely on someone remembering to click
# a button).
# ---------------------------------------------------------------------------
_last_cleanup_result = None

def run_cleanup():
    """Runs one cleanup pass using the currently configured retention
    windows. Safe to call from a request handler or a background thread:
    it does not touch anything the room-scoping/auth layer cares about,
    since it operates on the full, unscoped event/clip stores, the same
    as CameraWorker writing to them."""
    global _last_cleanup_result
    s = load_system_settings()
    retention_days = s.get("retention_days", 90)
    fp_retention_days = s.get("false_positive_retention_days", 7)

    # mutate_if (not mutate): this runs once a day against potentially
    # thousands of events and usually deletes nothing, so an
    # unconditional write here would mean a wasted full-file rewrite
    # every single day it finds nothing to clean up. Locked the same as
    # every other events.json writer, so a cleanup pass can no longer
    # race with, for example, add_event or an incident-notes save landing
    # mid-cleanup and having its change silently dropped.
    deleted_count = []
    def _mutate(events):
        kept, deleted = cleanup_events(events, retention_days, fp_retention_days)
        deleted_count.append(deleted)
        if deleted:
            events[:] = kept
            return True
        return False
    _events_store.mutate_if(_mutate)
    deleted_events = deleted_count[0]

    deleted_clips = cleanup_clips(CLIPS_DIR, retention_days)

    _last_cleanup_result = {
        "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deleted_events": deleted_events,
        "deleted_clips": deleted_clips,
        "retention_days": retention_days,
        "false_positive_retention_days": fp_retention_days,
    }
    return _last_cleanup_result


def start_retention_thread(interval_seconds=24 * 60 * 60):
    """Starts a daemon thread that runs cleanup once immediately, then
    once per interval_seconds. Call this explicitly from a real entry
    point (this file's __main__ block, or src/main.py), not at module
    import time, so importing dashboard.app (for example in tests, which
    reload this module fresh per test) never spawns a background thread
    that outlives the test and keeps touching a temp directory that is
    about to be torn down."""
    def _loop():
        while True:
            try:
                run_cleanup()
            except Exception as e:
                print(f"[RETENTION] Cleanup pass failed, will retry next interval: {e}")
            time.sleep(interval_seconds)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


@app.route("/api/admin/run-cleanup", methods=["POST"])
@admin_required
def trigger_cleanup():
    return jsonify({"status": "ok", **run_cleanup()})


@app.route("/api/admin/cleanup-status")
@admin_required
def cleanup_status():
    return jsonify(_last_cleanup_result or {"ran_at": None})


# ---------------------------------------------------------------------------
# Invites: admin-only. Creating an invite is the only way a new sign-up
# link comes into existence; nothing here lets a caller pick their own
# token or bypass the admin_required check.
# ---------------------------------------------------------------------------
@app.route("/api/invites")
@admin_required
def get_invites():
    return jsonify(list_pending_invites())

@app.route("/api/invites", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def add_invite():
    data = request.json or {}
    email = (data.get("email") or "").strip()
    name  = (data.get("name") or "").strip()
    role  = data.get("role", "caregiver")
    assigned_rooms = data.get("assigned_rooms") or []
    if not isinstance(assigned_rooms, list):
        assigned_rooms = []
    try:
        invite = create_invite(email, name, role, invited_by=current_user.email, assigned_rooms=assigned_rooms)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    link = url_for("signup", token=invite["token"], _external=True)
    return jsonify({
        "status": "ok",
        "link": link,
        "email": invite["email"],
        "role": invite["role"],
        "assigned_rooms": invite["assigned_rooms"],
        "expires_at": invite["expires_at"],
    })

@app.route("/api/invites/<token>", methods=["DELETE"])
@admin_required
def delete_invite(token):
    ok = revoke_invite(token)
    return jsonify({"status": "ok" if ok else "not found"}), (200 if ok else 404)

@app.route("/api/invites/by-email/<email>", methods=["DELETE"])
@admin_required
def delete_invite_by_email(email):
    ok = revoke_invite_by_email(email)
    return jsonify({"status": "ok" if ok else "not found"}), (200 if ok else 404)


# ---------------------------------------------------------------------------
# Caregiver room access: admin-only. Existing accounts (created before
# room scoping existed, or via the CLI without --rooms) default to
# seeing no rooms rather than every room; see auth/users.py's
# create_caregiver for why. This is how an admin fixes that for an
# account after the fact, without having to delete and re-invite them.
# ---------------------------------------------------------------------------
@app.route("/api/caregivers")
@admin_required
def get_caregivers():
    return jsonify(list_caregivers())

@app.route("/api/caregivers/<email>/rooms", methods=["POST"])
@admin_required
def update_caregiver_rooms(email):
    data = request.json or {}
    rooms = data.get("assigned_rooms")
    if not isinstance(rooms, list):
        return jsonify({"error": "assigned_rooms must be a list of room names"}), 400
    updated = set_assigned_rooms(email, rooms)
    if not updated:
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": "ok", "email": email, "assigned_rooms": updated.get("assigned_rooms", [])})


# ---------------------------------------------------------------------------
# System settings: read available to any caregiver, edits admin-only.
# ---------------------------------------------------------------------------
@app.route("/api/system-settings")
@login_required
def get_system_settings():
    s = load_system_settings()
    s["model_path"] = getattr(_cfg, "MODEL_PATH", "models/finetuned_model.pt") if _cfg else "models/finetuned_model.pt"
    return jsonify(s)

@app.route("/api/system-settings", methods=["POST"])
@admin_required
def update_system_settings():
    data = request.json or {}
    def _mutate(stored):
        # stored is the raw on-disk overrides dict (mutate() loaded it
        # under the lock); _system_settings_with_defaults folds in the
        # Config-derived defaults for anything not yet stored, using
        # THIS load rather than a fresh, unlocked load_system_settings()
        # call, so a concurrent update() landing in between can't have
        # its change silently overwritten by this one saving a
        # stale merged snapshot.
        s = _system_settings_with_defaults(stored)
        for key in ("confirm_seconds", "buffer_seconds", "post_event_seconds"):
            if key in data:
                s[key] = int(data[key])
        if "motion_threshold" in data:
            s["motion_threshold"] = float(data["motion_threshold"])
        if "retention_days" in data:
            s["retention_days"] = max(1, int(data["retention_days"]))
        if "false_positive_retention_days" in data:
            s["false_positive_retention_days"] = max(1, int(data["false_positive_retention_days"]))
        stored.clear()
        stored.update(s)
    _sys_settings_store.mutate(_mutate)
    return jsonify({"status": "ok"})

@app.route("/api/system-settings/upload-model", methods=["POST"])
@admin_required
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["model"]
    # f.filename is exactly what the client sent in the upload's
    # Content-Disposition header -- entirely attacker-controlled, and
    # ".pt" alone does nothing to stop "../../../whatever.pt". secure_filename()
    # strips path separators and ".." segments; the resolved-path check
    # below is a second, independent guard so this fails closed even if a
    # future Werkzeug version changes what secure_filename() considers
    # safe, rather than trusting a single library call to get it right.
    filename = secure_filename(f.filename or "")
    if not filename.endswith(".pt"):
        return jsonify({"error": "Model file must be a .pt file"}), 400

    dest = (MODELS_DIR / filename).resolve()
    if dest.parent != MODELS_DIR.resolve():
        return jsonify({"error": "Invalid filename"}), 400

    f.save(str(dest))

    def _mutate(stored):
        s = _system_settings_with_defaults(stored)
        s["model_path"] = f"models/{filename}"
        stored.clear()
        stored.update(s)
    _sys_settings_store.mutate(_mutate)
    return jsonify({
        "status": "ok",
        "note": "Model file saved. Update MODEL_PATH in config.py and restart the detection service to use it.",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    start_retention_thread()
    app.run(host="0.0.0.0", port=port, debug=False)
