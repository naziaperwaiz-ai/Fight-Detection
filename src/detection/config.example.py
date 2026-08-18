# src/detection/config.example.py
# Copy this file to config.py and fill in your values.
# NEVER commit config.py to GitHub.

class Config:
    MODEL_PATH         = "models/finetuned_model.pt"
    CAMERA_SOURCE      = 0
    MOTION_THRESHOLD   = 1.5
    VIOLENCE_THRESHOLD = 0.90
    CONFIRM_SECONDS    = 3
    FPS                = 15
    BUFFER_SECONDS     = 10
    POST_EVENT_SECONDS = 15
    CLIPS_DIR          = "outputs/clips"
    COOLDOWN_SECONDS   = 120
    EMAIL_SENDER       = "your-alert-email@gmail.com"
    EMAIL_APP_PASSWORD = "your-16-char-app-password"
    EMAIL_RECIPIENTS   = ["recipient@gmail.com"]
    ROOM_NAME          = "Ward A"
    CAMERA_ID          = "CAM-01"
    DASHBOARD_URL      = "http://localhost:5000"

    # --- Fall detection -------------------------------------------------
    # Independent of the violence pipeline above: a bbox-collapse +
    # sustained-horizontal rule (see detection/state_machine.py
    # StateMachine._update_fall). No training data or trained model
    # involved -- these thresholds are geometry, tune per camera angle.
    FALL_HEIGHT_DROP_RATIO = 0.5   # bbox height must drop to <= this fraction of "standing" height
    FALL_LOOKBACK_SECONDS  = 2.0   # window used to find the "standing" reference height
    FALL_CONFIRM_SECONDS   = 2.0   # must stay horizontal this long before confirming
    FALL_MIN_BBOX_HEIGHT   = 40    # px; boxes smaller than this are too noisy to judge

    # --- Hazard detection (dangerous object near a wrist) ---------------
    # Opt-in. Object detection (knife/scissors/fork) rides along on the
    # existing per-frame person-detection pass (near-free class filtering
    # on an already-running forward pass -- see detector.py's
    # extra_classes) rather than a second full detector model. Only the
    # pose model below is a separate network, and it auto-downloads its
    # weights on first use if not cached. See detection/hazard.py.
    # Rule-based, same reasoning as the fall rule -- no labelled data for
    # "holding a knife" exists in this project.
    HAZARD_DETECTION_ENABLED     = False
    HAZARD_POSE_WEIGHTS          = "yolov8n-pose.pt"
    HAZARD_IMGSZ                 = 320    # pose inference size; lower = faster, worse small-object recall
    HAZARD_SAMPLE_EVERY_N_FRAMES = 5      # throttle the pose model -- it's the still-expensive part
    HAZARD_PROXIMITY_FRAC        = 0.06   # wrist-to-object distance as a fraction of frame diagonal
    HAZARD_MIN_CONSECUTIVE       = 2      # consecutive samples before an event fires
    HAZARD_MIN_SEVERITY          = "high" # "high" = knife/scissors only; "low" also reports cutlery

    # --- Security -----------------------------------------------------
    # SECRET_KEY: signs caregiver login sessions. Generate a real random
    # value per deployment, e.g.:  python -c "import secrets; print(secrets.token_hex(32))"
    # Never reuse the example value below and never commit a real key.
    SECRET_KEY         = "change-me-to-a-random-64-char-hex-string"

    # INTERNAL_API_KEY: shared secret the detection pipeline uses to post
    # events to the dashboard's /api/events/add endpoint. This is separate
    # from caregiver login -- it authenticates server-to-server calls from
    # CameraWorker, which has no browser session to log in with. Generate
    # the same way as SECRET_KEY and keep it out of version control.
    INTERNAL_API_KEY   = "change-me-to-a-random-internal-key"