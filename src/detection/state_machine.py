# src/detection/state_machine.py
# Per-person state machine with 6 states.
# Each tracked person has independent state.

import time
import numpy as np


class PersonState:
    NORMAL    = "Normal"
    PROXIMATE = "Proximate"
    AGITATED  = "Agitated"
    FIGHTING  = "Fighting"
    ON_GROUND = "OnGround"
    EMERGENCY = "Emergency"


class PersonTrack:
    def __init__(self, track_id):
        self.track_id      = track_id
        self.state         = PersonState.NORMAL
        self.last_seen     = time.time()
        self.state_since   = time.time()
        self.bbox_history  = []   # last N bboxes for aspect ratio check
        self.score_history = []   # last N violence scores

        # Fall detection is intentionally a SEPARATE signal from `state`
        # above. `state` tracks violence escalation (Normal -> ... ->
        # Fighting -> OnGround -> Emergency), which requires a violence
        # score first -- a person who trips with no altercation never
        # enters that chain. fall_status is a parallel, independent rule:
        # "did this person's bounding box collapse in height and stay
        # horizontal", with no dependency on the violence classifier at
        # all. A person can be Normal (violence-wise) and FallConfirmed
        # (fall-wise) at the same time -- that's the correct behavior.
        self.height_history     = []   # (timestamp, bbox_height) samples, last few seconds
        self.fall_status        = "None"   # "None" | "Suspected" | "Confirmed"
        self.fall_status_since  = time.time()
        self.fall_recovery_since = None    # when they stopped being horizontal, while still Confirmed

    def time_in_state(self):
        return time.time() - self.state_since

    def set_state(self, new_state):
        if new_state != self.state:
            self.state       = new_state
            self.state_since = time.time()

    def update_bbox(self, bbox):
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 30:
            self.bbox_history.pop(0)

    def update_score(self, score):
        self.score_history.append(score)
        if len(self.score_history) > 15:
            self.score_history.pop(0)

    def avg_score(self):
        if not self.score_history:
            return 0.0
        return float(np.mean(self.score_history))

    def is_horizontal(self):
        """Check if bounding box is wider than tall — person on ground."""
        if not self.bbox_history:
            return False
        x1, y1, x2, y2 = self.bbox_history[-1]
        w = x2 - x1
        h = y2 - y1
        return w > h if h > 0 else False

    def update_height(self, bbox):
        """Record (timestamp, bbox_height) for the fall-collapse check.

        Kept separate from bbox_history (which is frame-count-bounded and
        used by is_horizontal/drawing) because the fall rule needs a
        time-bounded window regardless of frame rate.
        """
        x1, y1, x2, y2 = bbox
        now = time.time()
        self.height_history.append((now, y2 - y1))
        cutoff = now - 6.0   # generous: covers lookback + confirm + margin
        self.height_history = [(t, h) for t, h in self.height_history if t >= cutoff]

    def reference_standing_height(self, lookback_seconds):
        """Max bbox height seen in the recent window, excluding the last
        0.5s so a fall already in progress can't be used as its own
        "was standing" baseline."""
        now = time.time()
        candidates = [
            h for t, h in self.height_history
            if 0.5 < (now - t) <= lookback_seconds
        ]
        return max(candidates) if candidates else 0.0


class StateMachine:
    def __init__(self, config):
        self.cfg    = config
        self.tracks = {}   # track_id -> PersonTrack
        self.alerts = {}   # track_id -> last alert time

    def _get_or_create(self, track_id):
        if track_id not in self.tracks:
            self.tracks[track_id] = PersonTrack(track_id)
        return self.tracks[track_id]

    def _cleanup_old_tracks(self):
        """Remove tracks not seen for 5 seconds."""
        now     = time.time()
        to_del  = [tid for tid, t in self.tracks.items() if now - t.last_seen > 5]
        for tid in to_del:
            del self.tracks[tid]

    def _check_proximity(self, bboxes):
        """
        Returns list of (id1, id2) pairs that are close to each other.
        Proximity = bounding boxes overlap or centers within threshold.
        """
        proximate = []
        ids = list(bboxes.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1, id2 = ids[i], ids[j]
                b1 = bboxes[id1]
                b2 = bboxes[id2]
                # check if boxes overlap or are within 50px
                cx1 = (b1[0] + b1[2]) / 2
                cy1 = (b1[1] + b1[3]) / 2
                cx2 = (b2[0] + b2[2]) / 2
                cy2 = (b2[1] + b2[3]) / 2
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                avg_h = ((b1[3]-b1[1]) + (b2[3]-b2[1])) / 2
                if dist < avg_h * 1.5:  # within 1.5x person height
                    proximate.append((id1, id2))
        return proximate

    def _update_fall(self, track):
        """Rule-based fall detection, independent of the violence pipeline.

        Rule: bbox height collapses to <= FALL_HEIGHT_DROP_RATIO of its
        recent "standing" reference height, and the bbox stays wider-than-
        tall (is_horizontal) continuously for FALL_CONFIRM_SECONDS.

        This is deliberately the same style as hazard_detect.py: a plain
        geometric rule on top of the already-running person detector, not
        a trained classifier. There is no labelled fall dataset anywhere
        in this project (RLVS/SCVD/UCF-Crime are all violence clips), so
        there is nothing to validate precision/recall against -- treat
        fall_status as a triage signal, not a scored model output.

        Known false-positive sources: sitting down quickly, bending down,
        a pet or object crossing the detector, or a camera angle where
        "wider than tall" doesn't correspond to lying down (e.g. a
        near-overhead mount). Tune FALL_* config per camera placement.
        """
        cfg          = self.cfg
        drop_ratio   = getattr(cfg, "FALL_HEIGHT_DROP_RATIO", 0.5)
        lookback     = getattr(cfg, "FALL_LOOKBACK_SECONDS", 2.0)
        confirm_secs = getattr(cfg, "FALL_CONFIRM_SECONDS", 2.0)
        min_height   = getattr(cfg, "FALL_MIN_BBOX_HEIGHT", 40)
        recovery_secs = 3.0

        if not track.height_history:
            return
        now, cur_height = track.height_history[-1]

        # A too-small/noisy box can't be positively read as "collapsed" or
        # "still horizontal" -- but it must NOT short-circuit the whole
        # function, or a track stuck Confirmed (e.g. partially occluded
        # while down) would never reach the recovery branch below and
        # would re-alert forever. Treat "too small to judge" as "not
        # horizontal": it can't start a new Suspected fall, and it lets
        # an existing Confirmed fall decay back to None via the normal
        # recovery timer instead of freezing.
        too_small  = cur_height < min_height
        ref_height = 0.0 if too_small else track.reference_standing_height(lookback)
        collapsed  = (not too_small) and ref_height > 0 and cur_height <= ref_height * drop_ratio
        horizontal = (not too_small) and track.is_horizontal()

        if track.fall_status == "None":
            if collapsed and horizontal:
                track.fall_status       = "Suspected"
                track.fall_status_since = now

        elif track.fall_status == "Suspected":
            # Same brief-occlusion tolerance as the Confirmed branch below,
            # via the same fall_recovery_since field (the two states are
            # mutually exclusive, so reusing it is safe). Without this, one
            # noisy not-horizontal sample mid-window -- furniture or another
            # person briefly crossing the box -- hard-reset progress to
            # "None" and a genuine fall could never accumulate confirm_secs
            # of "mostly horizontal" time.
            if horizontal:
                track.fall_recovery_since = None
                if now - track.fall_status_since >= confirm_secs:
                    track.fall_status = "Confirmed"
            else:
                if track.fall_recovery_since is None:
                    track.fall_recovery_since = now
                elif now - track.fall_recovery_since > recovery_secs:
                    track.fall_status         = "None"
                    track.fall_recovery_since = None

        elif track.fall_status == "Confirmed":
            if horizontal:
                track.fall_recovery_since = None
            else:
                if track.fall_recovery_since is None:
                    track.fall_recovery_since = now
                elif now - track.fall_recovery_since > recovery_secs:
                    track.fall_status         = "None"
                    track.fall_recovery_since = None

    def update(self, track_id, score, bbox, frame):
        """Update state for one tracked person."""
        track = self._get_or_create(track_id)
        track.last_seen = time.time()
        track.update_bbox(bbox)
        track.update_score(score)
        track.update_height(bbox)
        self._update_fall(track)

        avg = track.avg_score()

        # state transitions
        if track.state == PersonState.NORMAL:
            if avg >= 0.4:
                track.set_state(PersonState.AGITATED)
            # proximity handled separately in update_all

        elif track.state == PersonState.PROXIMATE:
            if avg >= 0.5:
                track.set_state(PersonState.AGITATED)
            elif avg < 0.2 and track.time_in_state() > 5:
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.AGITATED:
            if avg >= self.cfg.VIOLENCE_THRESHOLD:
                track.set_state(PersonState.FIGHTING)
            elif avg < 0.3 and track.time_in_state() > 3:
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.FIGHTING:
            if track.is_horizontal():
                track.set_state(PersonState.ON_GROUND)
            elif avg < 0.3 and track.time_in_state() > 5:
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.ON_GROUND:
            if track.time_in_state() > 30:
                track.set_state(PersonState.EMERGENCY)
            elif not track.is_horizontal() and track.time_in_state() > 3:
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.EMERGENCY:
            # stays until manually reset
            pass

        self._cleanup_old_tracks()
        return track.state

    def update_all(self, detections):
        """
        Call after processing all detections in a frame.
        Checks proximity between all detected persons.
        """
        bboxes = {tid: bbox for tid, bbox, _ in detections}
        proximate_pairs = self._check_proximity(bboxes)

        for id1, id2 in proximate_pairs:
            t1 = self.tracks.get(id1)
            t2 = self.tracks.get(id2)
            if t1 and t1.state == PersonState.NORMAL:
                t1.set_state(PersonState.PROXIMATE)
            if t2 and t2.state == PersonState.NORMAL:
                t2.set_state(PersonState.PROXIMATE)

    def get_states(self):
        """Returns dict of track_id -> state for dashboard display."""
        return {tid: t.state for tid, t in self.tracks.items()}

    def has_emergency(self):
        return any(t.state == PersonState.EMERGENCY for t in self.tracks.values())

    def has_fighting(self):
        return any(t.state == PersonState.FIGHTING for t in self.tracks.values())

    def has_fall(self):
        return any(t.fall_status == "Confirmed" for t in self.tracks.values())