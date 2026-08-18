# src/dashboard/retention.py
#
# Age-based cleanup for incident records and saved clips. Two independent
# windows:
#   - retention_days: how long ANY incident record / clip file is kept.
#   - false_positive_retention_days: a shorter window for incidents
#     already marked false_positive. Confirmed noise doesn't need the
#     full retention window, and getting rid of it sooner reduces how
#     much patient video sits on disk with no ongoing purpose -- that's
#     the whole point of this file existing.
#
# Clip files and incident records are cleaned up independently, each by
# its own timestamp (clip file mtime; an incident's own "timestamp"
# field), not cross-referenced. That's a deliberate simplification: the
# alert event's clip_path is set to the placeholder string "Saving..."
# when the incident is first logged and is never retroactively updated
# once the real clip exists (a separate "Clip Ready" event carries the
# real path) -- so trying to precisely delete "this incident's clip"
# would be building on a link that's already unreliable elsewhere in
# this codebase, not something this cleanup pass should paper over.

import time
from datetime import datetime


def _event_age_days(event, now):
    try:
        t = datetime.strptime(event["timestamp"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError, TypeError):
        return None
    return (now - t).total_seconds() / 86400.0


def cleanup_events(events, retention_days, false_positive_retention_days, now=None):
    """Returns (kept_events, deleted_count).

    An event whose timestamp can't be parsed is kept, not deleted -- we
    don't guess an age for it, and erring toward keeping unparseable
    records is safer than erring toward silently destroying them.
    """
    now = now or datetime.now()
    kept = []
    deleted = 0
    for e in events:
        age = _event_age_days(e, now)
        if age is None:
            kept.append(e)
            continue
        limit = false_positive_retention_days if e.get("false_positive") else retention_days
        if age > limit:
            deleted += 1
        else:
            kept.append(e)
    return kept, deleted


def cleanup_clips(clips_dir, retention_days, now_ts=None):
    """Deletes .mp4 files in clips_dir whose mtime is older than
    retention_days. Returns the number deleted.
    """
    now_ts = time.time() if now_ts is None else now_ts
    deleted = 0
    for f in clips_dir.glob("*.mp4"):
        try:
            age_days = (now_ts - f.stat().st_mtime) / 86400.0
        except OSError:
            continue  # file removed between glob() and stat() -- fine, skip it
        if age_days > retention_days:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass  # already gone, or a permissions issue -- don't crash cleanup over one file
    return deleted
