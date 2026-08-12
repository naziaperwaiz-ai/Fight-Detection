import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path


class Notifier:
    def __init__(self, config):
        self.cfg = config

    def send_alert(self, camera_id: str, room: str, event_type: str,
                   confidence: float, clip_path: str):
        self._send_email(camera_id, room, event_type, confidence, clip_path)

    def _send_email(self, camera_id, room, event_type, confidence, clip_path):
        try:
            msg = MIMEMultipart()
            msg["From"]    = self.cfg.EMAIL_SENDER
            msg["To"]      = ", ".join(self.cfg.EMAIL_RECIPIENTS)
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
                    self.cfg.EMAIL_RECIPIENTS,
                    msg.as_string()
                )
            print(f"[EMAIL] Alert sent to {self.cfg.EMAIL_RECIPIENTS}")

        except Exception as e:
            print(f"[EMAIL ERROR] {e}")