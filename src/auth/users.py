# src/auth/users.py
#
# Minimal account store. Two roles exist: "caregiver" (default) and
# "admin". Admins can additionally change the detection model and the
# detection defaults from System Settings; caregivers see those as
# read-only. Accounts are pre-provisioned -- there is no self-registration
# route anywhere in the app. New accounts are created with
# `create_caregiver.py` (run by whoever administers the deployment), not
# through the web UI.
#
# Storage is a flat JSON file, matching how cameras.json/events.json
# already work in this project. Passwords are never stored in plaintext:
# only a salted hash (Werkzeug's PBKDF2-based hasher) is kept.
#
# If this ever grows past a handful of accounts, swap this for a real
# database -- but keep the interface (get_by_id / get_by_email /
# verify_caregiver) the same so callers in app.py don't need to change.

import json
import secrets
import time
import uuid
from pathlib import Path
from threading import Lock

from werkzeug.security import generate_password_hash, check_password_hash

_BASE = Path(__file__).parent.parent.parent
USERS_FILE = _BASE / "outputs" / "logs" / "users.json"
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Invites let an admin provision a new account without running the CLI --
# the admin issues a one-time, expiring, single-use token from the
# dashboard and hands the resulting link to the new hire. The invitee sets
# their own password; nobody can reach the sign-up form without a token an
# admin generated first, so this does not reopen self-registration.
INVITES_FILE = _BASE / "outputs" / "logs" / "invites.json"
INVITES_FILE.parent.mkdir(parents=True, exist_ok=True)
INVITE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

_lock = Lock()


def _load():
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return []


def _save(users):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def get_caregiver_by_id(user_id):
    for u in _load():
        if u["id"] == user_id:
            return u
    return None


def get_caregiver_by_email(email):
    email = (email or "").strip().lower()
    for u in _load():
        if u["email"] == email:
            return u
    return None


def verify_caregiver(email, password):
    """Return the user record if email+password are correct, else None.

    Deliberately returns the same thing (None) whether the email doesn't
    exist or the password is wrong -- callers must not use this to
    distinguish the two, so the login screen can't be used to enumerate
    valid caregiver emails.
    """
    record = get_caregiver_by_email(email)
    if not record:
        # Run the hash check anyway against a dummy hash so that a
        # non-existent email doesn't respond measurably faster than a
        # wrong password (basic timing-attack hygiene).
        check_password_hash(
            "pbkdf2:sha256:600000$dummy$dummy", password or ""
        )
        return None
    if not check_password_hash(record["password_hash"], password or ""):
        return None
    return record


def create_caregiver(email, password, name=None, role="caregiver", assigned_rooms=None):
    """assigned_rooms: list of room names this caregiver can view cameras,
    incidents, and clips for. Ignored for admins (admins always see every
    room -- see app.py's Caregiver.can_access_room). Defaults to [] --
    i.e. NO rooms -- for caregivers, on purpose: a newly-created caregiver
    account should not default to seeing every patient's video before an
    admin has deliberately assigned them a zone. This is a stricter
    default than most fields here, and that's intentional for anything
    that gates access to patient video.
    """
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if role not in ("caregiver", "admin"):
        raise ValueError("Role must be 'caregiver' or 'admin'.")

    with _lock:
        users = _load()
        if any(u["email"] == email for u in users):
            raise ValueError(f"An account with email {email} already exists.")
        record = {
            "id": uuid.uuid4().hex,
            "email": email,
            "name": name or email,
            "role": role,
            "assigned_rooms": list(assigned_rooms) if assigned_rooms else [],
            "password_hash": generate_password_hash(password),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        users.append(record)
        _save(users)
    return record


def set_assigned_rooms(email, rooms):
    """Admin-facing: update which rooms an existing caregiver can access.
    Returns the updated record, or None if no account has this email.
    """
    email = (email or "").strip().lower()
    rooms = [r for r in (rooms or []) if isinstance(r, str) and r.strip()]
    with _lock:
        users = _load()
        for u in users:
            if u["email"] == email:
                u["assigned_rooms"] = rooms
                _save(users)
                return u
    return None


def list_caregivers():
    """Return caregiver records without password hashes, for admin/CLI use."""
    return [
        {k: v for k, v in u.items() if k != "password_hash"} for u in _load()
    ]


def delete_caregiver(email):
    email = (email or "").strip().lower()
    with _lock:
        users = _load()
        remaining = [u for u in users if u["email"] != email]
        if len(remaining) == len(users):
            return False
        _save(remaining)
    return True


# ---------------------------------------------------------------------------
# Invites
# ---------------------------------------------------------------------------
def _load_invites():
    if INVITES_FILE.exists():
        return json.loads(INVITES_FILE.read_text())
    return []


def _save_invites(invites):
    INVITES_FILE.write_text(json.dumps(invites, indent=2))


def create_invite(email, name, role, invited_by, assigned_rooms=None):
    """Create a one-time invite. Only callable from an admin-only route.

    assigned_rooms: rooms the resulting caregiver account will be able to
    see (ignored for admin invites). Defaults to [] -- see create_caregiver
    for why that's the deliberate default, not an oversight.
    """
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if role not in ("caregiver", "admin"):
        raise ValueError("Role must be 'caregiver' or 'admin'.")
    if get_caregiver_by_email(email):
        raise ValueError(f"An account with email {email} already exists.")

    with _lock:
        invites = _load_invites()
        # Invalidate any older, still-pending invites for the same email so
        # only the newest link works.
        invites = [i for i in invites if i["email"] != email or i["used"]]
        now = time.time()
        record = {
            "token": secrets.token_urlsafe(32),
            "email": email,
            "name": name or email,
            "role": role,
            "assigned_rooms": list(assigned_rooms) if assigned_rooms else [],
            "invited_by": invited_by,
            "created_at": now,
            "expires_at": now + INVITE_TTL_SECONDS,
            "used": False,
        }
        invites.append(record)
        _save_invites(invites)
    return record


def get_invite(token):
    for i in _load_invites():
        if secrets.compare_digest(i["token"], token or ""):
            return i
    return None


def get_valid_invite(token):
    """Return the invite only if it exists, is unused, and hasn't expired."""
    invite = get_invite(token)
    if not invite:
        return None
    if invite["used"] or time.time() > invite["expires_at"]:
        return None
    return invite


def consume_invite(token, password):
    """Mark an invite used and create the account it names, atomically.

    Re-validates the invite inside the lock so two near-simultaneous
    submits of the same link can't both succeed.
    """
    with _lock:
        invites = _load_invites()
        invite = next(
            (i for i in invites if secrets.compare_digest(i["token"], token or "")),
            None,
        )
        if not invite or invite["used"] or time.time() > invite["expires_at"]:
            raise ValueError("This invite link is invalid, expired, or already used.")
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")

        users = _load()
        if any(u["email"] == invite["email"] for u in users):
            invite["used"] = True
            _save_invites(invites)
            raise ValueError("An account with this email already exists.")

        record = {
            "id": uuid.uuid4().hex,
            "email": invite["email"],
            "name": invite["name"],
            "role": invite["role"],
            "assigned_rooms": invite.get("assigned_rooms", []),
            "password_hash": generate_password_hash(password),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        users.append(record)
        _save(users)

        invite["used"] = True
        _save_invites(invites)
    return record


def list_pending_invites():
    """Unused, unexpired invites, for the admin UI. Never includes tokens."""
    now = time.time()
    return [
        {k: v for k, v in i.items() if k != "token"}
        for i in _load_invites()
        if not i["used"] and i["expires_at"] > now
    ]


def revoke_invite(token):
    with _lock:
        invites = _load_invites()
        remaining = [i for i in invites if not secrets.compare_digest(i["token"], token or "")]
        if len(remaining) == len(invites):
            return False
        _save_invites(remaining)
    return True


def revoke_invite_by_email(email):
    """Admin-facing revoke that doesn't require holding the token. Safe to
    expose to any admin_required route: the caller already had to
    authenticate as an admin to reach this, and email (unlike the token)
    isn't a bearer secret."""
    email = (email or "").strip().lower()
    with _lock:
        invites = _load_invites()
        remaining = [i for i in invites if i["email"] != email]
        if len(remaining) == len(invites):
            return False
        _save_invites(remaining)
    return True
