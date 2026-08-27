# src/detection/config.example.py
# Copy this file to config.py. Secrets (email credentials, SECRET_KEY,
# INTERNAL_API_KEY) are read from environment variables below rather than
# filled in as literal strings here -- see the --- Security --- section
# for why and how to set them. NEVER commit config.py to GitHub.
import os
import secrets as _secrets


def _env_or_generate(name):
    """Reads name from the environment; if unset, generates a fresh
    random value for this process only and warns loudly. See config.py's
    copy of this function for the full explanation."""
    value = os.environ.get(name)
    if value:
        return value
    generated = _secrets.token_hex(32)
    print(
        f"[CONFIG] {name} is not set in the environment; generated a "
        f"temporary value for this process only. Set the {name} "
        f"environment variable for a real deployment, or sessions and "
        f"the internal API key will reset on every restart."
    )
    return generated


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

    # Set HAVEN_EMAIL_SENDER / HAVEN_EMAIL_APP_PASSWORD /
    # HAVEN_EMAIL_RECIPIENTS (comma-separated) in the environment. No
    # generated fallback exists for these (a made-up password can't send
    # real mail); Notifier.send_alert fails soft and just skips sending
    # if they're blank, so the app still runs without them configured.
    EMAIL_SENDER       = os.environ.get("HAVEN_EMAIL_SENDER", "")
    EMAIL_APP_PASSWORD = os.environ.get("HAVEN_EMAIL_APP_PASSWORD", "")
    EMAIL_RECIPIENTS   = [
        r.strip() for r in os.environ.get("HAVEN_EMAIL_RECIPIENTS", "").split(",") if r.strip()
    ]
    ROOM_NAME          = "Ward A"
    CAMERA_ID          = "CAM-01"
    DASHBOARD_URL      = "http://localhost:5000"
    # DASHBOARD_CERT_PATH is not set here -- main.py sets it automatically
    # at startup (see its __main__ block) to whichever cert file the
    # dashboard is actually serving, if any, so CameraWorker's internal
    # alert POSTs (pipeline.py's _dispatch_alert) can verify against that
    # exact self-signed cert instead of failing SSL verification against
    # the public CA bundle. Nothing to configure here.

    # --- Fall detection -------------------------------------------------
    # Independent of the violence pipeline above: a bbox-collapse +
    # sustained-horizontal rule (see detection/state_machine.py
    # StateMachine._update_fall). No training data or trained model
    # involved; these thresholds are geometry, tune per camera angle.
    FALL_HEIGHT_DROP_RATIO = 0.5   # bbox height must drop to <= this fraction of "standing" height
    FALL_LOOKBACK_SECONDS  = 2.0   # window used to find the "standing" reference height
    FALL_CONFIRM_SECONDS   = 2.0   # must stay horizontal this long before confirming
    FALL_MIN_BBOX_HEIGHT   = 40    # px; boxes smaller than this are too noisy to judge

    # --- Hazard detection (dangerous object near a wrist) ---------------
    # Opt-in. Object detection (knife/scissors/fork) rides along on the
    # existing per-frame person-detection pass (near-free class filtering
    # on an already-running forward pass; see detector.py's
    # extra_classes) rather than a second full detector model. Only the
    # pose model below is a separate network, and it auto-downloads its
    # weights on first use if not cached. See detection/hazard.py.
    # Rule-based, same reasoning as the fall rule: no labelled data for
    # "holding a knife" exists in this project.
    HAZARD_DETECTION_ENABLED     = False
    HAZARD_POSE_WEIGHTS          = "yolov8n-pose.pt"
    HAZARD_IMGSZ                 = 320    # pose inference size; lower = faster, worse small-object recall
    HAZARD_SAMPLE_EVERY_N_FRAMES = 5      # throttle the pose model; it's the still-expensive part
    HAZARD_PROXIMITY_FRAC        = 0.06   # wrist-to-object distance as a fraction of frame diagonal
    HAZARD_MIN_CONSECUTIVE       = 2      # consecutive samples before an event fires
    HAZARD_MIN_SEVERITY          = "high" # "high" = knife/scissors only; "low" also reports cutlery

    # How long a red box + label stays drawn on a flagged hazard object
    # after it fires (see pipeline.py's _flag_hazard_box/_draw_hazard_boxes).
    # The object detector only re-locates hazard objects on throttled
    # samples (HAZARD_SAMPLE_EVERY_N_FRAMES), so this box is the object's
    # last known position held for a short window, not a live per-frame
    # tracked box -- same tradeoff the live-feed blur placeholder below
    # already makes.
    HAZARD_BOX_DISPLAY_SECONDS   = 5

    # Supervision context (see pipeline.py's process_frame): a hazard
    # object near a wrist while someone else is also in frame reads as
    # normal, expected activity -- cooking together, handing over
    # scissors -- not something a caregiver needs paged for. Set False
    # to alert on every hazard object regardless of who else is present.
    HAZARD_REQUIRE_UNSUPERVISED  = True
    # "Quiet hours" during which the patient is expected to be asleep --
    # an unsupervised hazard event during this window is marked with
    # elevated urgency (see _hazard_quiet_hours_active), though it does
    # not change whether the event fires. Both None (the default) means
    # quiet hours are off; there is no safe universal default here, it
    # depends on the patient's actual routine. 24-hour clock, wraps past
    # midnight correctly (e.g. 22 -> 6 means 10pm-6am).
    HAZARD_QUIET_HOURS_START     = None
    HAZARD_QUIET_HOURS_END       = None

    # --- Live feed privacy blur -----------------------------------------
    # The dashboard's live camera feed (video_feed) stays a frozen,
    # heavily pixelated placeholder except while a tracked person is at
    # Agitated or above, a fall is confirmed, or a hazard event just
    # fired; see pipeline.py's CameraWorker._publish_live_frame. This has
    # no admin override by design: there is no way to make a camera's
    # live feed show real video outside a trigger, only these two timing
    # knobs.
    LIVE_BLUR_HYSTERESIS_SECONDS = 15   # how long the feed stays unblurred after the last trigger
    LIVE_BLUR_REFRESH_SECONDS    = 10   # how often the frozen placeholder re-blurs from a fresh frame

    # --- Violence state machine thresholds ------------------------------
    # All of these have hardcoded fallback values in state_machine.py
    # (read via getattr(config, name, default), the same pattern as
    # FALL_*/HAZARD_* above), so nothing here is required -- only set
    # what you actually need to tune per deployment.
    STATE_AGITATED_SCORE               = 0.4   # Normal -> Agitated
    STATE_PROXIMATE_AGITATED_SCORE     = 0.5   # Proximate -> Agitated
    STATE_PROXIMATE_RECOVER_SCORE      = 0.2   # Proximate -> Normal (below this avg score...)
    STATE_PROXIMATE_RECOVER_SECONDS    = 5     # ...for at least this long
    STATE_AGITATED_RECOVER_SCORE       = 0.3   # Agitated -> Normal (below this avg score...)
    STATE_AGITATED_RECOVER_SECONDS     = 3     # ...for at least this long
    STATE_FIGHTING_RECOVER_SCORE       = 0.3   # Fighting -> Normal (below this avg score...)
    STATE_FIGHTING_RECOVER_SECONDS     = 5     # ...for at least this long
    STATE_ON_GROUND_EMERGENCY_SECONDS  = 30    # OnGround -> Emergency after this long still down
    STATE_ON_GROUND_RECOVER_SECONDS    = 3     # OnGround -> Normal, once sustained back on their feet
    STATE_EMERGENCY_RECOVER_SECONDS    = 5     # Emergency -> Normal, once sustained back on their feet
    # Consecutive same-reading frames required before a horizontal/
    # vertical bbox reading is trusted for a state transition (Fighting
    # -> OnGround, OnGround -> Normal, Emergency -> Normal). Debounces a
    # single noisy frame (a limb crossing the box, tracker jitter) out of
    # escalation/recovery decisions.
    STATE_HORIZONTAL_DEBOUNCE_FRAMES   = 3

    # Motion/proximity backup signal: an independent, score-free path to
    # Fighting for when the violence classifier under-scores real
    # violence (motion blur, an off-training-distribution camera angle).
    # Two proximate tracks that are BOTH moving fast relative to their
    # own bbox size, sustained for CONFIRM_SECONDS, escalate straight to
    # Fighting regardless of avg_score(). See
    # StateMachine._update_motion_fight_pair for the full rationale.
    STATE_MOTION_FIGHT_INTENSITY       = 1.2   # bbox-diagonals/sec displacement required, per person
    STATE_MOTION_FIGHT_CONFIRM_SECONDS = 1.5   # both must sustain that pace for this long to escalate
    STATE_MOTION_FIGHT_RECOVER_SECONDS = 1.0   # brief dip below the pace tolerated before progress resets
    # Both guard against a single real person's detector box being
    # briefly duplicated into two overlapping tracks (a tracker glitch,
    # most likely to happen exactly while someone is moving quickly) --
    # without these, that looks identical to "two proximate people both
    # moving fast" and falsely escalates one person alone to Fighting.
    STATE_MOTION_FIGHT_MIN_TRACK_AGE_SECONDS = 1.0   # a track younger than this can't count as "a second person" yet
    STATE_MOTION_FIGHT_MAX_IOU               = 0.3   # boxes overlapping more than this are treated as one body, not two

    # --- Security -----------------------------------------------------
    # SECRET_KEY: signs caregiver login sessions.
    # INTERNAL_API_KEY: shared secret the detection pipeline uses to post
    # events to the dashboard's /api/events/add endpoint (separate from
    # caregiver login; it authenticates server-to-server calls from
    # CameraWorker, which has no browser session to log in with).
    # Set both via environment variables for any deployment meant to stay
    # up across restarts, e.g.:
    #   export HAVEN_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    #   export HAVEN_INTERNAL_API_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    # If left unset, a random value is generated for this process only
    # (see _env_or_generate above) -- fine for a quick local check, but
    # every restart then invalidates all sessions and briefly breaks
    # CameraWorker's posts to the dashboard until both processes are
    # restarted together.
    SECRET_KEY         = _env_or_generate("HAVEN_SECRET_KEY")
    INTERNAL_API_KEY   = _env_or_generate("HAVEN_INTERNAL_API_KEY")