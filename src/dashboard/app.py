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
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import json, os, cv2, time, secrets, hmac, uuid, re

from auth.users import (
    verify_caregiver, get_caregiver_by_id,
    create_invite, get_valid_invite, consume_invite,
    list_pending_invites, revoke_invite, revoke_invite_by_email,
)

app = Flask(__name__)

_BASE = Path(__file__).parent.parent.parent
CLIPS_DIR       = _BASE / "outputs" / "clips"
MODELS_DIR      = _BASE / "models"
LOGS_FILE       = _BASE / "outputs" / "logs" / "events.json"
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
# point storage_uri at a shared Redis instance instead -- in-memory limits
# are per-process and won't be shared across workers, which would let an
# attacker get more attempts than intended just by hitting a different
# worker each time.
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

    @property
    def is_admin(self):
        return self.role == "admin"


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
    a clean 403 -- the frontend also hides these controls, but the backend
    check is what actually enforces it (never trust the UI alone)."""
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

def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def load_events():
    return _read_json(LOGS_FILE, [])

def save_events(events):
    _write_json(LOGS_FILE, events)

def load_cameras():
    # Default to an empty list, not a canned "CAM-01" seed -- a seed camera
    # here would silently persist once a real camera gets added on top of
    # it (add_camera loads-then-appends), leaving a fake camera in the list
    # forever. A fresh install should show zero cameras until you add one.
    return _read_json(CAMERAS_FILE, [])

def save_cameras(cameras):
    _write_json(CAMERAS_FILE, cameras)

def load_profiles():
    return _read_json(PROFILES_FILE, {})

def save_profiles(profiles):
    _write_json(PROFILES_FILE, profiles)

def load_announcements():
    existing = _read_json(ANNOUNCE_FILE, None)
    if existing is not None:
        return existing
    # Seed with example shift notes on first run so the panel isn't empty.
    seeded = [
        {"time": datetime.now().strftime("%H:%M"), "text": "Shift handover complete.", "icon": "icon-shift-handover.jpg", "author": "system"},
    ]
    save_announcements(seeded)
    return seeded

def save_announcements(items):
    _write_json(ANNOUNCE_FILE, items)

def load_settings():
    return _read_json(SETTINGS_FILE, {
        "recipients": [], "threshold": 90, "cooldown": 120, "email_channel": True,
    })

def save_settings(s):
    _write_json(SETTINGS_FILE, s)

def load_system_settings():
    defaults = {"confirm_seconds": 3, "motion_threshold": 1.5, "buffer_seconds": 10, "post_event_seconds": 15}
    if _cfg is not None:
        defaults = {
            "confirm_seconds": getattr(_cfg, "CONFIRM_SECONDS", defaults["confirm_seconds"]),
            "motion_threshold": getattr(_cfg, "MOTION_THRESHOLD", defaults["motion_threshold"]),
            "buffer_seconds": getattr(_cfg, "BUFFER_SECONDS", defaults["buffer_seconds"]),
            "post_event_seconds": getattr(_cfg, "POST_EVENT_SECONDS", defaults["post_event_seconds"]),
        }
    stored = _read_json(SYS_SETTINGS_FILE, None)
    if stored:
        defaults.update(stored)
    return defaults

def save_system_settings(s):
    _write_json(SYS_SETTINGS_FILE, s)


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
# There is intentionally no route that lets a visitor create an account by
# just showing up here -- that would undo the access control the caregiver
# login was built for. The only way to reach a working sign-up form is to
# hold a token an admin generated via POST /api/invites (see below). The
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
    return Response(generate_frames(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/score/<cam_id>")
@login_required
def get_score(cam_id):
    worker = _workers.get(cam_id)
    return jsonify({"score": worker.score if worker else 0.0})

@app.route("/api/me")
@login_required
def get_me():
    return jsonify({"id": current_user.id, "email": current_user.email, "name": current_user.name, "role": current_user.role})


# ---------------------------------------------------------------------------
# Events / incidents
# ---------------------------------------------------------------------------
@app.route("/api/events")
@login_required
def get_events():
    return jsonify(load_events()[-200:])

@app.route("/api/events/add", methods=["POST"])
@internal_key_required
@limiter.limit("60 per minute")
def add_event():
    # Called by CameraWorker (server-to-server) via the internal API key,
    # not by a caregiver's browser session.
    data   = request.json or {}
    events = load_events()
    events.append({
        "id":         uuid.uuid4().hex[:8].upper(),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id":  data.get("camera_id", "unknown"),
        "room":       data.get("room", "unknown"),
        "event_type": data.get("event_type", "unknown"),
        "confidence": data.get("confidence", 0.0),
        "clip_path":  data.get("clip_path", ""),
        "states":     data.get("states", []),  # snapshot of tracked-person states at alert time
        "notes":      "",
        "reviewed":   False,
        "false_positive": False,
    })
    save_events(events)
    return jsonify({"status": "ok"})

@app.route("/api/incidents/<incident_id>")
@login_required
def get_incident(incident_id):
    for e in load_events():
        if e.get("id") == incident_id:
            return jsonify(e)
    return jsonify({"error": "not found"}), 404

@app.route("/api/incidents/<incident_id>/notes", methods=["POST"])
@login_required
def set_incident_notes(incident_id):
    data = request.json or {}
    events = load_events()
    for e in events:
        if e.get("id") == incident_id:
            e["notes"] = data.get("notes", "")
            save_events(events)
            return jsonify({"status": "ok"})
    return jsonify({"error": "not found"}), 404

@app.route("/api/incidents/<incident_id>/review", methods=["POST"])
@login_required
def toggle_incident_review(incident_id):
    events = load_events()
    for e in events:
        if e.get("id") == incident_id:
            e["reviewed"] = not e.get("reviewed", False)
            save_events(events)
            return jsonify({"status": "ok", "reviewed": e["reviewed"]})
    return jsonify({"error": "not found"}), 404

@app.route("/api/incidents/<incident_id>/false-positive", methods=["POST"])
@login_required
def toggle_incident_false_positive(incident_id):
    events = load_events()
    for e in events:
        if e.get("id") == incident_id:
            e["false_positive"] = not e.get("false_positive", False)
            save_events(events)
            return jsonify({"status": "ok", "false_positive": e["false_positive"]})
    return jsonify({"error": "not found"}), 404


# ---------------------------------------------------------------------------
# Cameras -- any caregiver can view status and the live preview, but adding,
# editing, or deleting a camera is admin-only. Which physical cameras exist
# is a facility-configuration decision, not a day-to-day caregiving one, and
# a caregiver account being able to silently remove coverage from a room
# (accidentally or otherwise) with no confirmation and no audit trail was an
# oversight from before the caregiver/admin role split existed, not a
# deliberate choice -- closing it now.
# ---------------------------------------------------------------------------
@app.route("/api/cameras")
@login_required
def get_cameras():
    cameras = load_cameras()
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
    data    = request.json or {}
    cameras = load_cameras()
    cameras.append({
        "id":        data.get("id", f"CAM-{len(cameras)+1:02d}"),
        "room":      data.get("room", "Unknown Room"),
        "source":    data.get("source", "0"),
        "threshold": float(data.get("threshold", 0.7)),
        "active":    True,
        "patients":  int(data.get("patients", 0)),
        "priority":  data.get("priority", "Medium"),
    })
    save_cameras(cameras)
    return jsonify({"status": "ok"})

@app.route("/api/cameras/delete/<cam_id>", methods=["DELETE"])
@admin_required
def delete_camera(cam_id):
    cameras = [c for c in load_cameras() if c["id"] != cam_id]
    save_cameras(cameras)
    return jsonify({"status": "ok"})

@app.route("/api/cameras/update/<cam_id>", methods=["POST"])
@admin_required
def update_camera(cam_id):
    data    = request.json or {}
    cameras = load_cameras()
    for c in cameras:
        if c["id"] == cam_id:
            c.update(data)
    save_cameras(cameras)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
_CLIP_NAME_RE = re.compile(r"^alert_(?P<cam>.+)_(?P<ts>\d{8}_\d{6})\.mp4$")

@app.route("/clips/<filename>")
@login_required
def serve_clip(filename):
    return send_from_directory(str(CLIPS_DIR.absolute()), filename)

@app.route("/api/clips")
@login_required
def get_clips():
    clips = sorted(CLIPS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for c in clips[:100]:
        m = _CLIP_NAME_RE.match(c.name)
        out.append({
            "filename": c.name,
            "camera_id": m.group("cam") if m else "unknown",
            "timestamp": m.group("ts") if m else "",
            "mtime": c.stat().st_mtime,
        })
    return jsonify(out)


# ---------------------------------------------------------------------------
# Per-caregiver profile (row-level scoping: always current_user.id, never a
# client-supplied id -- see the note on /api/me in the design doc).
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
    profiles = load_profiles()
    profiles[current_user.id] = {
        "about": data.get("about", ""),
        "notes": data.get("notes", ""),
    }
    save_profiles(profiles)
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
    items = load_announcements()
    items.append({
        "time": datetime.now().strftime("%H:%M"),
        "text": text[:500],
        "author": current_user.name,
        "icon": None,
    })
    save_announcements(items)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Alert settings (any caregiver can view/edit -- matches how the physical
# on-call sheet already works: whoever's on shift can update it)
# ---------------------------------------------------------------------------
@app.route("/api/settings")
@login_required
def get_settings():
    return jsonify(load_settings())

@app.route("/api/settings", methods=["POST"])
@login_required
def update_settings():
    data = request.json or {}
    s = load_settings()
    if "recipients" in data:
        s["recipients"] = [r for r in data["recipients"] if isinstance(r, str) and "@" in r]
    if "threshold" in data:
        s["threshold"] = max(50, min(100, int(data["threshold"])))
    if "cooldown" in data:
        s["cooldown"] = max(30, min(300, int(data["cooldown"])))
    if "email_channel" in data:
        s["email_channel"] = bool(data["email_channel"])
    save_settings(s)
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
    events = load_events()
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
# Invites -- admin-only. Creating an invite is the only way a new sign-up
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
    try:
        invite = create_invite(email, name, role, invited_by=current_user.email)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    link = url_for("signup", token=invite["token"], _external=True)
    return jsonify({
        "status": "ok",
        "link": link,
        "email": invite["email"],
        "role": invite["role"],
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
# System settings -- read available to any caregiver, edits admin-only.
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
    s = load_system_settings()
    for key in ("confirm_seconds", "buffer_seconds", "post_event_seconds"):
        if key in data:
            s[key] = int(data[key])
    if "motion_threshold" in data:
        s["motion_threshold"] = float(data["motion_threshold"])
    save_system_settings(s)
    return jsonify({"status": "ok"})

@app.route("/api/system-settings/upload-model", methods=["POST"])
@admin_required
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["model"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Model file must be a .pt file"}), 400
    dest = MODELS_DIR / f.filename
    f.save(str(dest))
    s = load_system_settings()
    s["model_path"] = f"models/{f.filename}"
    save_system_settings(s)
    return jsonify({
        "status": "ok",
        "note": "Model file saved. Update MODEL_PATH in config.py and restart the detection service to use it.",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
