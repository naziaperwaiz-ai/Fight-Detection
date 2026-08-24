# tests/test_notifier.py
#
# Unit tests for notification/notifier.py's live-settings wiring: a
# caregiver's saved Alert Settings (recipients, email_channel) must
# actually change what a real alert does, not just what
# /api/settings/test-alert uses. Covers:
#   - load_alert_settings(): reads the real file, and degrades to {} on
#     a missing or unreadable file rather than raising.
#   - Notifier.send_alert(): honors a live email_channel=False by not
#     sending at all, and prefers live recipients over the Config
#     default, falling back to the Config default when the live
#     settings have no recipients.
#
#   PYTHONPATH=src pytest tests/test_notifier.py -v

import json
import smtplib
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from notification import notifier as notifier_mod  # noqa: E402
from notification.notifier import Notifier, load_alert_settings  # noqa: E402


def _cfg(recipients=("cfg@example.com",)):
    return SimpleNamespace(
        EMAIL_SENDER="sender@example.com",
        EMAIL_APP_PASSWORD="not-a-real-password",
        EMAIL_RECIPIENTS=list(recipients),
    )


def test_load_alert_settings_reads_the_real_file(tmp_path, monkeypatch):
    settings_file = tmp_path / "alert_settings.json"
    settings_file.write_text(json.dumps({"recipients": ["a@example.com"], "email_channel": False}))
    monkeypatch.setattr(notifier_mod, "_ALERT_SETTINGS_FILE", settings_file)
    assert load_alert_settings() == {"recipients": ["a@example.com"], "email_channel": False}


def test_load_alert_settings_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(notifier_mod, "_ALERT_SETTINGS_FILE", tmp_path / "does_not_exist.json")
    assert load_alert_settings() == {}


def test_load_alert_settings_corrupt_file_returns_empty_dict(tmp_path, monkeypatch):
    settings_file = tmp_path / "alert_settings.json"
    settings_file.write_text("{not valid json")
    monkeypatch.setattr(notifier_mod, "_ALERT_SETTINGS_FILE", settings_file)
    assert load_alert_settings() == {}


def test_send_alert_skips_entirely_when_email_channel_is_off(monkeypatch):
    monkeypatch.setattr(notifier_mod, "load_alert_settings", lambda: {"email_channel": False})

    def _fail_if_called(*a, **k):
        raise AssertionError("_send_email must not be called when email_channel is off")
    monkeypatch.setattr(Notifier, "_send_email", _fail_if_called)

    Notifier(_cfg()).send_alert("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4")


def test_send_alert_uses_live_recipients_over_config_default(monkeypatch):
    monkeypatch.setattr(
        notifier_mod, "load_alert_settings",
        lambda: {"email_channel": True, "recipients": ["live@example.com"]},
    )
    captured = {}

    def _capture(self, camera_id, room, event_type, confidence, clip_path, recipients):
        captured["recipients"] = recipients
    monkeypatch.setattr(Notifier, "_send_email", _capture)

    Notifier(_cfg(recipients=("cfg@example.com",))).send_alert("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4")
    assert captured["recipients"] == ["live@example.com"]


def test_send_alert_falls_back_to_config_recipients_when_live_settings_have_none(monkeypatch):
    # No settings file yet, or one saved with an empty recipients list:
    # either way, a real alert must still go to whoever config.py has,
    # rather than silently emailing nobody.
    monkeypatch.setattr(notifier_mod, "load_alert_settings", lambda: {})
    captured = {}

    def _capture(self, camera_id, room, event_type, confidence, clip_path, recipients):
        captured["recipients"] = recipients
    monkeypatch.setattr(Notifier, "_send_email", _capture)

    Notifier(_cfg(recipients=("cfg@example.com",))).send_alert("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4")
    assert captured["recipients"] == ["cfg@example.com"]


# ---------------------------------------------------------------------
# Blank-credential handling: config.example.py documents
# "Notifier.send_alert fails soft and just skips sending if [sender/app
# password] are blank" -- that used to not actually be true. With no
# check for this, a blank sender/password fell straight through to
# _send_email() and produced a real SMTP auth error (535, "Username and
# Password not accepted") indistinguishable in the console from
# wrong-but-present credentials, instead of the documented graceful
# skip with a message saying credentials were never configured at all.
# ---------------------------------------------------------------------

def test_send_alert_skips_when_sender_is_blank(monkeypatch, capsys):
    monkeypatch.setattr(notifier_mod, "load_alert_settings", lambda: {"email_channel": True})

    def _fail_if_called(*a, **k):
        raise AssertionError("_send_email must not be called with a blank sender")
    monkeypatch.setattr(Notifier, "_send_email", _fail_if_called)

    cfg = _cfg()
    cfg.EMAIL_SENDER = ""
    Notifier(cfg).send_alert("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4")

    assert "Skipped" in capsys.readouterr().out


def test_send_alert_skips_when_app_password_is_blank(monkeypatch, capsys):
    monkeypatch.setattr(notifier_mod, "load_alert_settings", lambda: {"email_channel": True})

    def _fail_if_called(*a, **k):
        raise AssertionError("_send_email must not be called with a blank app password")
    monkeypatch.setattr(Notifier, "_send_email", _fail_if_called)

    cfg = _cfg()
    cfg.EMAIL_APP_PASSWORD = ""
    Notifier(cfg).send_alert("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4")

    assert "Skipped" in capsys.readouterr().out


def test_send_alert_still_sends_when_credentials_are_present(monkeypatch):
    # Regression guard against the blank-credential check above being
    # too eager and skipping a real, fully-configured send.
    monkeypatch.setattr(notifier_mod, "load_alert_settings", lambda: {"email_channel": True})
    called = {"n": 0}
    monkeypatch.setattr(Notifier, "_send_email", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    Notifier(_cfg()).send_alert("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4")
    assert called["n"] == 1


def test_send_email_gives_a_specific_message_on_gmail_auth_rejection(monkeypatch, capsys):
    # Regression test: smtplib's raw SMTPAuthenticationError (535,
    # "Username and Password not accepted") gives no hint that this
    # almost always means "not a Google App Password" or "2-Step
    # Verification isn't turned on yet" -- both easy to fix once named,
    # opaque otherwise.
    def _raise_auth_error(self, sender, password):
        raise smtplib.SMTPAuthenticationError(
            535, b"5.7.8 Username and Password not accepted."
        )

    class _FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, sender, password):
            _raise_auth_error(self, sender, password)
        def sendmail(self, *a, **k): pass

    monkeypatch.setattr(notifier_mod.smtplib, "SMTP_SSL", _FakeSMTP)

    Notifier(_cfg())._send_email("CAM-01", "Room A", "Violence Detected", 0.9, "clip.mp4", ["a@example.com"])

    out = capsys.readouterr().out
    assert "App Password" in out
    assert "2-Step Verification" in out
