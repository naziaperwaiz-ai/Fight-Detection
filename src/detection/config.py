# src/detection/config.py
# All tunable parameters in one place.
# Swap model by changing MODEL_PATH.
# Swap camera by changing CAMERA_SOURCE.

class Config:
    # ── Model ──────────────────────────────────────────────
    MODEL_PATH = "models/finetuned_model.pt"   # drop any .pt file here to swap

    # ── Camera ─────────────────────────────────────────────
    CAMERA_SOURCE = 0          # 0 = laptop webcam
                               # "rtsp://..." = IP camera
                               # "path/to/video.mp4" = test video file
    MOTION_THRESHOLD = 1.5  # tune this — higher = less sensitive to motion

    # ── Detection ──────────────────────────────────────────
    VIOLENCE_THRESHOLD = 0.90   # avg score above this triggers alert
    CONFIRM_SECONDS    = 3     # must persist this long before alerting
    FPS                = 15    # expected camera FPS

    # ── Recording ──────────────────────────────────────────
    BUFFER_SECONDS     = 10    # pre-event buffer length
    POST_EVENT_SECONDS = 15    # how long to record after alert
    CLIPS_DIR          = "outputs/clips"

    # ── Notifications ──────────────────────────────────────
    COOLDOWN_SECONDS   = 120   # min seconds between alerts per camera

    # ── Notifications ──────────────────────────────────────────
    EMAIL_SENDER       = "fightdetectionalerts@gmail.com"
    EMAIL_APP_PASSWORD = "eyldoxixbhyevhzt"
    EMAIL_RECIPIENTS   = ["muhammaduzair32323@gmail.com"]
    ROOM_NAME          = "Ward A"
    CAMERA_ID          = "CAM-01"
    DASHBOARD_URL = "http://localhost:5000"