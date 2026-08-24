import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path

# Same directory depth as dashboard/app.py's _BASE (src/dashboard/app.py's
# parent.parent.parent), so this resolves to the same repo root and the
# same file, without importing dashboard.app (a Flask app) into the
# detection/notification layer.
_ALERT_SETTINGS_FILE = Path(__file__).parent.parent.parent / "outputs" / "logs" / "alert_settings.json"


def load_alert_settings():
    """Reads the caregiver-facing Alert Settings (recipients, threshold,
    cooldown, email_channel, ...) directly from the JSON file the
    dashboard writes, so a setting a caregiver saves in the UI actually
    changes what a running detection process does, not just what the
    next test-alert call uses. Returns {} on a missing file, an
    unreadable file, or any other read error, so every caller falls back
    to its own Config default rather than crashing on a settings file
    that has not been created yet or is mid-write.
    """
    try:
        return json.loads(_ALERT_SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class Notifier:
    def __init__(self, config):
        self.cfg = config

    def send_alert(self, camera_id: str, room: str, event_type: str,
                   confidence: float, clip_path: str):
        settings = load_alert_settings()
        if not settings.get("email_channel", True):
            print(f"[EMAIL] Skipped: email channel is turned off in Alert Settings")
            return
        if not self.cfg.EMAIL_SENDER or not self.cfg.EMAIL_APP_PASSWORD:
            # config.example.py documents this as "fails soft and just
            # skips sending if they're blank" -- that was previously not
            # actually true: with no check here, a blank sender/password
            # fell straight through to _send_email() and produced a real
            # SMTP auth error (535, "Username and Password not
            # accepted") indistinguishable in the console from wrong-but-
            # present credentials. Checked here instead, with a message
            # that says which of the two problems it actually is, so a
            # caregiver isn't left guessing whether they need to fix a
            # typo'd app password or set one up for the first time.
            print("[EMAIL] Skipped: HAVEN_EMAIL_SENDER/HAVEN_EMAIL_APP_PASSWORD "
                  "not set in the environment. No email will be sent until both "
                  "are configured (see config.example.py's Security/email section).")
            return
        recipients = settings.get("recipients") or list(self.cfg.EMAIL_RECIPIENTS)
        self._send_email(camera_id, room, event_type, confidence, clip_path, recipients)

    def _send_email(self, camera_id, room, event_type, confidence, clip_path, recipients):
        try:
            msg = MIMEMultipart()
            msg["From"]    = self.cfg.EMAIL_SENDER
            msg["To"]      = ", ".join(recipients)
            msg["Subject"] = f"[ALERT] {event_type} detected — {room}"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"""
VIOLENCE DETECTION ALERT
━━━━━━━━━━━━━━━━━━━━━━━━

Time:       {timestamp}
Room:       {room}
Camera:     {camera_id}
Event:      {event_type}
Confidence: {confidence:.1%}
Clip:       {clip_path}

This is an automated alert from the Fight Detection System.
Please respond immediately.
            """
            msg.attach(MIMEText(body, "plain"))

            # attach clip if it exists
            clip = Path(clip_path)
            if clip.exists() and clip.stat().st_size > 0:
                with open(clip, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={clip.name}"
                )
                msg.attach(part)
                print(f"[EMAIL] Attaching clip: {clip.name}")

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(self.cfg.EMAIL_SENDER, self.cfg.EMAIL_APP_PASSWORD)
                server.sendmail(
                    self.cfg.EMAIL_SENDER,
                    recipients,
                    msg.as_string()
                )
            print(f"[EMAIL] Alert sent to {recipients}")

        except smtplib.SMTPAuthenticationError as e:
            # Gmail's SMTP rejects the login itself here (535, "Username
            # and Password not accepted") -- this is Gmail refusing the
            # credentials, not a bug in this code, but the raw SMTPlib
            # exception gives no hint of the two most common causes, so
            # spell them out: EMAIL_APP_PASSWORD must be a Google App
            # Password (myaccount.google.com/apppasswords), never the
            # account's normal login password, which Gmail has not
            # accepted for SMTP in years; and an App Password can only be
            # generated once 2-Step Verification is turned on for that
            # Google account, so a Gmail account without 2FA cannot use
            # this at all until it is enabled.
            print(f"[EMAIL ERROR] Gmail rejected the login ({e}). Most likely cause: "
                  "EMAIL_APP_PASSWORD is not a Google App Password (a regular Gmail "
                  "password will not work), or the account does not have 2-Step "
                  "Verification turned on yet, which is required before an App "
                  "Password can even be generated. Generate one at "
                  "myaccount.google.com/apppasswords and set it as "
                  "HAVEN_EMAIL_APP_PASSWORD.")
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")