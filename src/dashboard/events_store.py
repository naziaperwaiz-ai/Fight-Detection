# src/dashboard/events_store.py
#
# SQLite-backed replacement for events.json (Incident History). Matches
# JsonStore's own load/save/mutate/mutate_if surface exactly (see
# dashboard/app.py), so every existing caller -- add_event,
# get_incident, the notes/review/false-positive toggles, retention's
# run_cleanup -- keeps working completely unchanged. Only what happens
# underneath a mutate() call is different.
#
# Why this exists: events.json was a single JSON file, read in full and
# rewritten in full on every single mutation -- one new alert, one
# notes edit, one review toggle, one nightly retention pass. That's
# fine for one or two cameras with occasional alerts, but does not
# scale cleanly: every additional camera means more writers serializing
# on that one lock, and each mutation still pays the cost of rewriting
# the *entire* incident history just to change a single row. SQLite
# keeps the same "one file, nothing extra to install or run" deployment
# model (still just a file on disk, no separate database server), while
# replacing whole-file rewrites with real per-row INSERT/UPDATE/DELETE
# statements against an indexed table instead of an unindexed Python
# list.
#
# mutate(fn)/mutate_if(fn) still hand `fn` a plain list of event dicts
# (the same shape load() has always returned) to mutate in place --
# callers do not need to know or care that storage changed underneath
# them. After fn returns, that list is diffed against a snapshot taken
# before fn ran, and only the actual changes (a new id -> INSERT, an id
# missing from the result -> DELETE, an existing id with changed fields
# -> UPDATE) are applied in one transaction, instead of a full-table
# rewrite.
#
# Existing events.json data is migrated automatically, once, the first
# time this store is actually used (not at construction -- see
# _ensure_conn) if it finds an events.json but an empty/nonexistent
# database. The original file is renamed to events.json.migrated, not
# deleted, so nothing is destroyed if a migration ever looks wrong.

import json
import sqlite3
import threading
import uuid
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id             TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    camera_id      TEXT NOT NULL DEFAULT 'unknown',
    room           TEXT NOT NULL DEFAULT 'unknown',
    event_type     TEXT NOT NULL DEFAULT 'unknown',
    confidence     REAL NOT NULL DEFAULT 0,
    clip_path      TEXT NOT NULL DEFAULT '',
    states         TEXT NOT NULL DEFAULT '[]',
    detail         TEXT NOT NULL DEFAULT '',
    notes          TEXT NOT NULL DEFAULT '',
    reviewed       INTEGER NOT NULL DEFAULT 0,
    false_positive INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_room ON events(room);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""

# Every column except id, in the fixed order used for INSERT/UPDATE
# statements below.
_VALUE_COLUMNS = [
    "timestamp", "camera_id", "room", "event_type", "confidence",
    "clip_path", "states", "detail", "notes", "reviewed", "false_positive",
]
_ALL_COLUMNS = ["id"] + _VALUE_COLUMNS

_DEFAULTS = {
    "timestamp": "", "camera_id": "unknown", "room": "unknown",
    "event_type": "unknown", "confidence": 0.0, "clip_path": "",
    "states": [], "detail": "", "notes": "", "reviewed": False,
    "false_positive": False,
}


def _new_id():
    # Matches add_event's own id shape (uuid4 hex, 8 chars, uppercase).
    # Only used as a fallback for an event dict that somehow has no id
    # at all -- should never happen via the real add_event endpoint,
    # but a test or a future caller building a dict by hand might omit
    # one, the same way JsonStore never required an id either.
    return uuid.uuid4().hex[:8].upper()


def _row_to_event(row):
    e = dict(row)
    e["states"] = json.loads(e["states"]) if e["states"] else []
    e["reviewed"] = bool(e["reviewed"])
    e["false_positive"] = bool(e["false_positive"])
    return e


def _event_to_row_values(event):
    """Builds the {column: value} dict for one event, filling in the
    same defaults callers throughout app.py have always relied on via
    `.get(key, default)` (add_event's own construction, _visible_events,
    retention's cleanup_events) -- so a partial dict, such as a test
    seeding events directly via save_events() with only a few fields
    set, is stored the same way JsonStore would have stored it
    verbatim, just with the gaps made explicit here instead of only
    appearing as a .get() default wherever the field is later read."""
    row = {}
    for col in _VALUE_COLUMNS:
        row[col] = event.get(col, _DEFAULTS[col])
    row["states"] = json.dumps(row["states"])
    row["reviewed"] = 1 if row["reviewed"] else 0
    row["false_positive"] = 1 if row["false_positive"] else 0
    return row


class SqliteEventsStore:
    """Same load/save/mutate/mutate_if surface as JsonStore (see
    dashboard/app.py), backed by SQLite instead of a single JSON file.
    See module docstring for the full rationale."""

    def __init__(self, db_path, legacy_json_path=None):
        self.db_path = Path(db_path)
        self.legacy_json_path = Path(legacy_json_path) if legacy_json_path else None
        self._lock = threading.Lock()
        # Opened lazily on first real use, not here -- see _ensure_conn.
        # dashboard/app.py builds this store at module import time,
        # before test fixtures get a chance to unlink/isolate the
        # on-disk path (see tests/test_dashboard.py's app_client
        # fixture); eagerly connecting and running the legacy-JSON
        # migration in __init__ would touch real files during import,
        # before that isolation applies. JsonStore has the identical
        # property: it never reads its file until the first
        # .load()/.mutate() call either.
        self._conn = None

    def _ensure_conn_locked(self):
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        if self.legacy_json_path is not None:
            self._migrate_legacy_json_if_needed_locked()

    def _migrate_legacy_json_if_needed_locked(self):
        """One-time import of an existing events.json into this table.
        Only runs when the table is genuinely empty AND a legacy file
        exists, so a restart after the first successful migration never
        re-imports (or overwrites real data already in the database).
        """
        legacy_path = self.legacy_json_path
        if not legacy_path.exists():
            return
        existing = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if existing:
            return
        try:
            legacy_events = json.loads(legacy_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(legacy_events, list) or not legacy_events:
            return
        print(f"[EVENTS] Migrating {len(legacy_events)} incident(s) from "
              f"{legacy_path.name} into {self.db_path.name}...")
        for event in legacy_events:
            self._insert_one_locked(event)
        self._conn.commit()
        migrated_path = legacy_path.with_name(legacy_path.name + ".migrated")
        legacy_path.rename(migrated_path)
        print(f"[EVENTS] Migration complete. {legacy_path.name} renamed to "
              f"{migrated_path.name} (kept, not deleted).")

    def _load_locked(self):
        rows = self._conn.execute(
            f"SELECT {', '.join(_ALL_COLUMNS)} FROM events ORDER BY rowid"
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def _insert_one_locked(self, event):
        """Inserts one new row. Does not commit -- callers batch their
        own commit so a multi-row operation (a full replace, a
        mutate()'s diff) is one transaction, not one fsync per row."""
        eid = event.get("id") or _new_id()
        event["id"] = eid
        values = _event_to_row_values(event)
        columns = ", ".join(_ALL_COLUMNS)
        placeholders = ", ".join("?" for _ in _ALL_COLUMNS)
        self._conn.execute(
            f"INSERT INTO events ({columns}) VALUES ({placeholders})",
            [eid] + [values[c] for c in _VALUE_COLUMNS],
        )

    def _update_one_locked(self, eid, event):
        values = _event_to_row_values(event)
        set_clause = ", ".join(f"{c} = ?" for c in _VALUE_COLUMNS)
        self._conn.execute(
            f"UPDATE events SET {set_clause} WHERE id = ?",
            [values[c] for c in _VALUE_COLUMNS] + [eid],
        )

    def _replace_all_locked(self, events):
        self._conn.execute("DELETE FROM events")
        for event in events:
            self._insert_one_locked(event)
        self._conn.commit()

    def _apply_diff_locked(self, events, originals):
        """events: the (possibly fn-mutated) list, post-fn.
        originals: {id: deep-copied pre-fn snapshot} for whatever was
        in the table when this mutate()/mutate_if() call started."""
        seen_ids = set()
        for event in events:
            eid = event.get("id") or _new_id()
            event["id"] = eid
            seen_ids.add(eid)
            if eid not in originals:
                self._insert_one_locked(event)
            elif event != originals[eid]:
                self._update_one_locked(eid, event)
        removed_ids = set(originals) - seen_ids
        if removed_ids:
            self._conn.executemany(
                "DELETE FROM events WHERE id = ?", [(i,) for i in removed_ids]
            )
        self._conn.commit()

    def load(self):
        with self._lock:
            self._ensure_conn_locked()
            return self._load_locked()

    def save(self, events):
        """Full replace, matching JsonStore.save()'s "this is now the
        entire contents" semantics exactly -- used directly by at least
        one test (save_events() seeding arbitrary partial event dicts),
        so this must accept dicts missing most fields, the same as
        _event_to_row_values' defaults handle everywhere else."""
        with self._lock:
            self._ensure_conn_locked()
            self._replace_all_locked(events)

    def mutate(self, fn):
        with self._lock:
            self._ensure_conn_locked()
            events = self._load_locked()
            originals = {e["id"]: dict(e) for e in events if e.get("id")}
            result = fn(events)
            self._apply_diff_locked(events, originals)
            return result

    def mutate_if(self, fn):
        with self._lock:
            self._ensure_conn_locked()
            events = self._load_locked()
            originals = {e["id"]: dict(e) for e in events if e.get("id")}
            if fn(events):
                self._apply_diff_locked(events, originals)
            return events


# ---------------------------------------------------------------------------
# Round check-ins ("I've walked through my assigned zones") -- a second
# table in the same events.db file, not a third storage mechanism next
# to SqliteEventsStore above and the handful of small JsonStore files
# dashboard/app.py still uses for cameras/profiles/announcements/
# settings. Unlike incident events, check-in volume does not scale with
# camera count -- it scales with caregiver headcount and shift cadence,
# a caregiver tapping one button a few times per shift -- so it did not
# independently justify a SQLite migration the way incident events did
# (see this module's docstring above). It moved here anyway to avoid
# accumulating a separate JSON file per feature when one database file
# with multiple tables does the same job with less on-disk sprawl.
#
# Check-ins are add-only from the app's perspective (nothing edits or
# deletes one after the fact), so this intentionally does not reuse
# SqliteEventsStore's diff-based mutate()/mutate_if() machinery, which
# exists specifically to support incident notes/review/false-positive
# edits after the fact. A plain add()/last_for()/list_all() surface is
# the right amount of abstraction for what this data actually needs.
# ---------------------------------------------------------------------------

_CHECKINS_SCHEMA = """
CREATE TABLE IF NOT EXISTS round_checkins (
    id             TEXT PRIMARY KEY,
    caregiver_id   TEXT NOT NULL,
    caregiver_name TEXT NOT NULL,
    rooms          TEXT NOT NULL,
    timestamp      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkins_caregiver ON round_checkins(caregiver_id);
CREATE INDEX IF NOT EXISTS idx_checkins_timestamp ON round_checkins(timestamp);
"""


def _row_to_checkin(row):
    c = dict(row)
    c["rooms"] = json.loads(c["rooms"])
    return c


class SqliteCheckinsStore:
    """Round check-in log, backed by a `round_checkins` table in the same
    SQLite file SqliteEventsStore uses (see module docstring above for
    why this isn't its own file or a JSON store). Lazily connects on
    first use for the same reason SqliteEventsStore does -- see that
    class's __init__ for the full explanation (dashboard/app.py builds
    stores at import time, before test fixtures get a chance to isolate
    the on-disk path)."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = None

    def _ensure_conn_locked(self):
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_CHECKINS_SCHEMA)
        conn.commit()
        self._conn = conn

    def add(self, record):
        """record: a fully-formed dict (id/caregiver_id/caregiver_name/
        rooms/timestamp already set by the caller -- see app.py's
        add_checkin, which builds it the same way add_event builds an
        event dict before handing it to SqliteEventsStore). Returns the
        same record back, matching add_event's own "return what got
        stored" convention."""
        with self._lock:
            self._ensure_conn_locked()
            self._conn.execute(
                "INSERT INTO round_checkins (id, caregiver_id, caregiver_name, rooms, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record["id"], record["caregiver_id"], record["caregiver_name"],
                    json.dumps(record["rooms"]), record["timestamp"],
                ),
            )
            self._conn.commit()
            return record

    def last_for(self, caregiver_id):
        """Most recent check-in for one caregiver, or None. Row-scoped by
        caregiver_id the same way /api/profile and this store's own
        add() are scoped by session identity in app.py -- never a
        client-supplied filter."""
        with self._lock:
            self._ensure_conn_locked()
            row = self._conn.execute(
                "SELECT id, caregiver_id, caregiver_name, rooms, timestamp FROM round_checkins "
                "WHERE caregiver_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT 1",
                (caregiver_id,),
            ).fetchone()
            return _row_to_checkin(row) if row else None

    def list_all(self, limit=200):
        """Every caregiver's check-ins, most recent first -- the admin
        audit view (see app.py's get_checkin_history, admin-only)."""
        with self._lock:
            self._ensure_conn_locked()
            rows = self._conn.execute(
                "SELECT id, caregiver_id, caregiver_name, rooms, timestamp FROM round_checkins "
                "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [_row_to_checkin(r) for r in rows]
