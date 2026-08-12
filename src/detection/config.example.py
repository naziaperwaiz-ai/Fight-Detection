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