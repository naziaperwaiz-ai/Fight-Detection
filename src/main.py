# src/main.py
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(__file__))

from detection.config import Config
from detection.multi_camera import MultiCameraEngine
from dashboard.app import app, register_worker, load_cameras, start_retention_thread, load_system_settings


def _build_camera_config(cam_config, dashboard_url=None, dashboard_cert_path=None):
    cfg                     = Config()
    cfg.CAMERA_ID           = cam_config["id"]
    cfg.ROOM_NAME           = cam_config["room"]
    cfg.CAMERA_SOURCE       = cam_config["source"]
    cfg.VIOLENCE_THRESHOLD  = float(cam_config.get("threshold", 0.7))
    # CameraWorker._dispatch_alert POSTs every fired alert (violence,
    # fighting, fall, hazard) to f"{DASHBOARD_URL}/api/events/add" -- the
    # only thing that ever writes an incident into events.json. Config's
    # class default is a bare "http://...", but once a TLS cert is
    # present (src/certs/generate_cert.py; see main.py's _ssl_context),
    # app.run() serves HTTPS-only on that port, and Flask's dev server
    # does not also speak plain HTTP there. A plain-http POST to an
    # https-only port fails to connect outright, and _dispatch_alert
    # swallows that error silently (by design, so a dashboard hiccup
    # never crashes detection) -- so every alert fires, the email/print
    # side still happens, and every single one simply vanishes instead
    # of ever reaching Incident History, with nothing in the console to
    # say so. dashboard_url is threaded in from __main__, where the
    # actual scheme actually being served is already known, rather than
    # re-deriving it here from a guess.
    if dashboard_url:
        cfg.DASHBOARD_URL = dashboard_url
    # requests.post (CameraWorker._dispatch_alert) defaults to verifying
    # the server cert against the public CA bundle, which will always
    # reject a self-signed cert -- so once DASHBOARD_URL is https://,
    # every internal alert POST was failing at the SSL handshake step,
    # silently (the surrounding try/except swallowed it with no log
    # line), meaning alerts fired and printed to the console but never
    # once reached Incident History. dashboard_cert_path is the exact
    # cert file the dashboard itself is serving (see main.py's
    # _ssl_context/__main__), passed to requests' `verify` param so it
    # trusts precisely that known cert instead of either failing
    # outright or disabling verification wholesale. None when serving
    # plain HTTP (verify is irrelevant for a non-https URL then).
    if dashboard_cert_path:
        cfg.DASHBOARD_CERT_PATH = dashboard_cert_path
    # Per-camera hazard opt-in from the dashboard's camera record (see
    # dashboard/app.py's add_camera). Everything else hazard-related
    # (severity, pose weights, image size, sample rate, proximity,
    # debounce length) still comes from the Config class defaults in
    # detection/config.py, shared by every camera; only whether hazard
    # detection runs at all is set per camera here. See
    # detection/multi_camera.py's module docstring for why the other
    # settings are necessarily shared in the batched engine.
    cfg.HAZARD_DETECTION_ENABLED = bool(cam_config.get("hazard_enabled", False))

    # System Settings (confirm_seconds/motion_threshold/buffer_seconds/
    # post_event_seconds), edited from the dashboard's System Settings
    # page and persisted to system_settings.json, used to have no effect
    # on a running engine at all: load_system_settings() was only ever
    # read back by the dashboard UI itself, never consulted here, so
    # saving a new value there silently did nothing to actual detection
    # behavior. These are shared across every camera (matching how they
    # are presented in the UI -- one set of values, not per-camera), so
    # they are applied here rather than per-camera like threshold/hazard
    # above.
    sys_settings = load_system_settings()
    cfg.CONFIRM_SECONDS    = sys_settings.get("confirm_seconds", cfg.CONFIRM_SECONDS)
    cfg.MOTION_THRESHOLD   = sys_settings.get("motion_threshold", cfg.MOTION_THRESHOLD)
    cfg.BUFFER_SECONDS     = sys_settings.get("buffer_seconds", cfg.BUFFER_SECONDS)
    cfg.POST_EVENT_SECONDS = sys_settings.get("post_event_seconds", cfg.POST_EVENT_SECONDS)
    # Hazard supervision context -- same "edited in System Settings,
    # applied here at engine startup, not live-reloaded" pattern as the
    # four settings just above. Previously these two only existed as
    # config.py class attributes with no dashboard control at all; see
    # _system_settings_with_defaults in dashboard/app.py for where the
    # defaults come from when nothing's been saved yet.
    # getattr, not direct attribute access: cfg here can be a user's own
    # config.py (or, in tests, a minimal stand-in), predating these
    # settings existing at all -- same defensive pattern pipeline.py
    # already uses for every HAZARD_* field, for the same reason.
    cfg.HAZARD_REQUIRE_UNSUPERVISED = sys_settings.get(
        "hazard_require_unsupervised", getattr(cfg, "HAZARD_REQUIRE_UNSUPERVISED", True))
    cfg.HAZARD_QUIET_HOURS_START = sys_settings.get(
        "hazard_quiet_hours_start", getattr(cfg, "HAZARD_QUIET_HOURS_START", None))
    cfg.HAZARD_QUIET_HOURS_END = sys_settings.get(
        "hazard_quiet_hours_end", getattr(cfg, "HAZARD_QUIET_HOURS_END", None))
    return cfg


def start_engine(camera_records, dashboard_url=None, dashboard_cert_path=None):
    """Builds one Config per registered camera and drives all of them
    through a single MultiCameraEngine: one shared model, detector, and
    hazard-pose-model instance, with one batched inference call per
    cycle, instead of one CameraWorker owning and calling its own full
    model copy per camera (see detection/multi_camera.py for why that
    does not scale past a handful of cameras).

    Each camera still gets registered with the dashboard as before
    (register_worker). get_frame()/score on the returned CameraWorker
    objects are unchanged, so /video_feed, /api/score, and the
    "Live"/"Offline" status all keep working with no dashboard changes
    needed.

    Only cameras with active=True get a worker started. dashboard/app.py's
    get_cameras() already computes liveStatus from whether a worker is
    registered for a camera id, so a camera marked inactive must not get
    a worker in the first place; otherwise the dashboard would still show
    "Live" (and /video_feed would still stream) for a camera an admin
    turned off, and the inactive flag would only ever affect its own
    display label instead of actually stopping detection for it.
    """
    active_records = [cam for cam in camera_records if cam.get("active", True)]
    skipped = len(camera_records) - len(active_records)
    if skipped:
        print(f"[MAIN] Skipping {skipped} inactive camera(s), not starting a worker for them.")
    if not active_records:
        # MultiCameraEngine requires at least one config (see its
        # __init__). Every registered camera being inactive is a normal
        # state, not an error, so this returns None the same way a
        # completely empty cameras.json does, instead of letting that
        # ValueError propagate.
        print("[MAIN] No active cameras to start (all registered cameras are inactive).")
        return None
    configs = [
        _build_camera_config(cam, dashboard_url=dashboard_url, dashboard_cert_path=dashboard_cert_path)
        for cam in active_records
    ]
    engine  = MultiCameraEngine(configs)
    for cam_id, worker in engine.workers.items():
        register_worker(cam_id, worker)
    engine.start()
    print(f"[MAIN] Engine started for {len(configs)} camera(s): "
          f"{', '.join(engine.workers.keys())}")
    return engine


def _ssl_context():
    """
    Look for a self-signed cert generated by src/certs/generate_cert.py.
    Serving over HTTPS is what lets the browser run secure-context APIs
    (Notification, etc). See src/certs/README.md for how to generate one
    and why a self-signed cert is the right tool for a facility's local
    network specifically, not for open-internet deployment.

    Paths can be overridden via HAVEN_SSL_CERT / HAVEN_SSL_KEY env vars,
    for example to point at a real cert from a reverse proxy setup
    instead. Falls back to plain HTTP (returns None) if no cert is
    present, so a fresh checkout still runs without extra setup; it just
    will not have secure-context browser features until a cert is
    generated.
    """
    certs_dir = os.path.join(os.path.dirname(__file__), "certs")
    cert_path = os.environ.get("HAVEN_SSL_CERT", os.path.join(certs_dir, "cert.pem"))
    key_path = os.environ.get("HAVEN_SSL_KEY", os.path.join(certs_dir, "key.pem"))
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return (cert_path, key_path)
    print(
        "[MAIN] No TLS cert found at "
        f"{cert_path}, serving over plain HTTP.\n"
        "[MAIN] Desktop notifications and other secure-context browser "
        "features won't work until you generate one: "
        "python src/certs/generate_cert.py"
    )
    return None


def run_dashboard(port, ssl_context):
    scheme = "https" if ssl_context else "http"
    print(f"[MAIN] Dashboard starting at {scheme}://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, ssl_context=ssl_context)


if __name__ == "__main__":
    # Resolved once, here, and threaded through to both run_dashboard
    # (so it serves the scheme this decided on) and start_engine below
    # (so CameraWorker's internal alert POSTs target that same scheme --
    # see _build_camera_config's dashboard_url comment). Calling
    # _ssl_context() a second time later would risk it disagreeing with
    # itself (e.g. printing the "no cert found" warning twice, or -- if
    # a cert happened to appear in between -- one caller deciding HTTPS
    # while the other still targets HTTP) for no benefit over deciding
    # once and reusing it.
    port        = int(os.environ.get("PORT", 5000))
    ssl_context = _ssl_context()
    scheme      = "https" if ssl_context else "http"
    dashboard_url = f"{scheme}://localhost:{port}"
    # The exact cert file the dashboard is about to serve (ssl_context is
    # (cert_path, key_path) when one exists), so CameraWorker's internal
    # alert POSTs can verify against precisely that cert instead of
    # either failing every HTTPS request's SSL verification against the
    # public CA bundle, or disabling verification altogether. See
    # _build_camera_config's dashboard_cert_path comment.
    dashboard_cert_path = ssl_context[0] if ssl_context else None

    # start dashboard
    t = threading.Thread(target=run_dashboard, args=(port, ssl_context), daemon=True)
    t.start()
    time.sleep(2)

    # Age-based cleanup of old incident records and clip files. Runs
    # once now, then once a day. See dashboard/retention.py and the
    # retention_days / false_positive_retention_days System Settings.
    start_retention_thread()

    # Start one shared-model engine covering every camera in
    # cameras.json (see MultiCameraEngine). A fresh install with no
    # cameras registered yet is a normal, expected state, for example
    # right after first startup, before an admin has added one through
    # the dashboard, so skip starting the engine rather than erroring on
    # an empty camera list. Cameras added afterward take effect on the
    # next restart, matching the behavior before this refactor (see
    # add_camera()'s comment in dashboard/app.py).
    cameras = load_cameras()
    if cameras:
        engine = start_engine(cameras, dashboard_url=dashboard_url, dashboard_cert_path=dashboard_cert_path)
        if engine is not None:
            print(f"[MAIN] {len(engine.workers)} camera(s) running. Press Ctrl+C to stop.")
        # start_engine already prints its own message when every
        # registered camera is inactive; nothing further to say here.
    else:
        engine = None
        print("[MAIN] No cameras registered in cameras.json yet. Dashboard is "
              "running, but no detection engine was started. Add a camera from "
              "the dashboard, then restart to pick it up.")

    # keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MAIN] Shutting down.")
        if engine is not None:
            engine.stop()
