# src/detection/state_machine.py
# Per-person state machine with 6 states.
# Each tracked person has independent state.

import time
from collections import deque

import numpy as np


def _bbox_iou(b1, b2):
    """Standard intersection-over-union between two (x1, y1, x2, y2)
    boxes, 0.0 if they don't overlap. Used by _update_motion_fight_pair
    to tell "two separate, overlapping people" apart from "one person's
    detector box, briefly duplicated" -- a duplicate almost completely
    overlaps its source box, which two distinct human bodies essentially
    never do even mid-grapple."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter   = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class PersonState:
    NORMAL    = "Normal"
    PROXIMATE = "Proximate"
    AGITATED  = "Agitated"
    FIGHTING  = "Fighting"
    ON_GROUND = "OnGround"
    EMERGENCY = "Emergency"


class PersonTrack:
    def __init__(self, track_id, score_window=15):
        self.track_id      = track_id
        self.state         = PersonState.NORMAL
        self.last_seen     = time.time()
        self.state_since   = time.time()
        self.first_seen    = time.time()  # see _update_motion_fight_pair's track-age guard
        self.bbox_history  = deque(maxlen=30)          # last N bboxes for aspect ratio check
        self.score_history = deque(maxlen=score_window)  # last N violence scores; FPS-scaled, see StateMachine._get_or_create

        # Fall detection is intentionally a separate signal from `state`
        # above. `state` tracks violence escalation (Normal -> ... ->
        # Fighting -> OnGround -> Emergency), which requires a violence
        # score first, so a person who trips with no altercation never
        # enters that chain. fall_status is a parallel, independent rule:
        # whether this person's bounding box collapsed in height and
        # stayed horizontal, with no dependency on the violence
        # classifier at all. A person can be Normal (violence-wise) and
        # FallConfirmed (fall-wise) at the same time; that is the
        # correct behavior.
        self.height_history     = deque()   # (timestamp, bbox_height) samples, last few seconds
        self.fall_status        = "None"   # "None" | "Suspected" | "Confirmed"
        self.fall_status_since  = time.time()
        self.fall_recovery_since = None    # when they stopped being horizontal, while still Confirmed

        # Consecutive-frame counters backing is_horizontal_sustained /
        # is_vertical_sustained below. A single noisy frame (a limb
        # crossing the box, a tracker jitter) should not by itself flip
        # a state transition that depends on "is this person on the
        # ground" -- these count consecutive same-answer frames instead
        # of trusting update_bbox's latest sample alone.
        self._horizontal_run = 0
        self._vertical_run   = 0

        # Motion/proximity backup signal: an independent, geometry-only
        # corroborating path to Fighting for cases where the violence
        # classifier under-scores real violence (e.g. motion blur, an
        # unfamiliar camera angle) and a track's avg_score() never even
        # clears STATE_AGITATED_SCORE. Deliberately mirrors _update_fall's
        # Suspected-with-recovery-tolerance shape: a rolling short window
        # of position samples, a sustained-confirm timer, and a brief-
        # interruption tolerance so one still frame doesn't reset
        # progress. See StateMachine._update_motion_fight_pair.
        self.position_history        = deque()  # (timestamp, cx, cy, diag) samples, last ~2s
        self._motion_fight_since     = None      # when sustained high motion started
        self._motion_fight_recovery_since = None  # when motion dropped, while still counting
        self.motion_confirmed_fight  = False      # True if this track's Fighting state came from motion, not score

    def time_in_state(self):
        return time.time() - self.state_since

    def set_state(self, new_state):
        if new_state != self.state:
            self.state       = new_state
            self.state_since = time.time()
            if new_state in (PersonState.NORMAL, PersonState.PROXIMATE, PersonState.AGITATED):
                # Only meaningful while a track is (or was) Fighting via
                # the motion path; clear it on any de-escalation so a
                # later, unrelated Fighting entry (e.g. via score) isn't
                # mislabeled as motion-confirmed.
                self.motion_confirmed_fight = False

    def update_bbox(self, bbox):
        self.bbox_history.append(bbox)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if h > 0 and w > h:
            self._horizontal_run += 1
            self._vertical_run    = 0
        else:
            self._vertical_run   += 1
            self._horizontal_run  = 0

    def update_score(self, score):
        self.score_history.append(score)

    def avg_score(self):
        if not self.score_history:
            return 0.0
        return float(np.mean(self.score_history))

    def is_horizontal(self):
        """Check if bounding box is wider than tall, meaning person on ground."""
        if not self.bbox_history:
            return False
        x1, y1, x2, y2 = self.bbox_history[-1]
        w = x2 - x1
        h = y2 - y1
        return w > h if h > 0 else False

    def is_horizontal_sustained(self, min_frames=3):
        """True once the box has read wider-than-tall for min_frames
        consecutive samples in a row (see update_bbox). Debounces
        single-frame noise out of state transitions that key off
        "person is on the ground"."""
        return self._horizontal_run >= min_frames

    def is_vertical_sustained(self, min_frames=3):
        """Mirror of is_horizontal_sustained for "back on their feet"
        transitions."""
        return self._vertical_run >= min_frames

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
        while self.height_history and self.height_history[0][0] < cutoff:
            self.height_history.popleft()

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

    def update_position(self, bbox):
        """Record (timestamp, center_x, center_y, bbox_diagonal) for the
        motion-fight backup signal.

        Kept separate from bbox_history/height_history for the same
        reason update_height is separate: this needs a short,
        time-bounded window regardless of frame rate, and a different
        cutoff (2s, tuned to how fast a real scuffle's motion shows up)
        than the fall rule's 6s lookback.
        """
        x1, y1, x2, y2 = bbox
        now  = time.time()
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        diag = float(np.hypot(x2 - x1, y2 - y1))
        self.position_history.append((now, cx, cy, diag))
        cutoff = now - 2.0
        while self.position_history and self.position_history[0][0] < cutoff:
            self.position_history.popleft()

    def motion_intensity(self, window_seconds=1.0):
        """Scale-invariant displacement rate over the recent window, in
        bbox-diagonals-per-second.

        Diagonal-normalizing the raw pixel displacement means this reads
        the same for a person close to the camera (large bbox, large
        pixel movement) and one further away (small bbox, small pixel
        movement) making the same real-world motion -- a plain pixels/sec
        threshold would only fire for whichever is closer to the lens.
        Uses the oldest and newest samples within window_seconds rather
        than summing frame-to-frame deltas, so a little tracker jitter
        back and forth doesn't inflate the reading.
        """
        now = time.time()
        recent = [s for s in self.position_history if now - s[0] <= window_seconds]
        if len(recent) < 2:
            return 0.0
        t0, cx0, cy0, diag0 = recent[0]
        t1, cx1, cy1, diag1 = recent[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        avg_diag = (diag0 + diag1) / 2
        if avg_diag <= 0:
            return 0.0
        dist = float(np.hypot(cx1 - cx0, cy1 - cy0))
        return (dist / avg_diag) / dt


class StateMachine:
    def __init__(self, config):
        self.cfg    = config
        self.tracks = {}   # track_id -> PersonTrack
        self.alerts = {}   # track_id -> last alert time

        # Per-pair sustained-motion timers for _update_motion_fight_pair,
        # keyed by a sorted (id1, id2) tuple. Must be per-pair, not a
        # single shared timer: a room can have more than one proximate
        # pair at once (e.g. 4 people, two separate scuffles), and each
        # pair's "how long has this been sustained" progress is
        # independent of every other pair's.
        self._motion_fight_pairs = {}   # (id1, id2) -> {"since": t|None, "recovery_since": t|None}

    def _get_or_create(self, track_id):
        if track_id not in self.tracks:
            # Score window scales with FPS the same way pipeline.py's
            # alert-confirmation window does (CONFIRM_SECONDS * FPS), so
            # avg_score() reflects the same wall-clock span regardless of
            # camera frame rate instead of a hardcoded frame count that
            # means ~1s at 15fps but ~0.5s at 30fps.
            confirm_seconds = getattr(self.cfg, "CONFIRM_SECONDS", 1)
            fps              = getattr(self.cfg, "FPS", 15)
            score_window     = max(1, int(confirm_seconds * fps))
            self.tracks[track_id] = PersonTrack(track_id, score_window=score_window)
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

        O(n^2) over the people visible in one frame. Left unchanged
        deliberately: this compares tracked people in a single room's
        camera view, realistically single digits to low tens of people,
        so the quadratic cost is negligible versus the per-frame
        detector/classifier inference cost. Revisit only if a deployment
        target puts many dozens of people in one camera's frame.
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
        there is nothing to validate precision/recall against. Treat
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

        # A too-small or noisy box cannot be positively read as
        # "collapsed" or "still horizontal", but it must not
        # short-circuit the whole function, or a track stuck Confirmed
        # (for example, partially occluded while down) would never reach
        # the recovery branch below and would re-alert forever. Treating
        # "too small to judge" as "not horizontal" means it cannot start
        # a new Suspected fall, and lets an existing Confirmed fall decay
        # back to None via the normal recovery timer instead of freezing.
        too_small  = cur_height < min_height
        ref_height = 0.0 if too_small else track.reference_standing_height(lookback)
        collapsed  = (not too_small) and ref_height > 0 and cur_height <= ref_height * drop_ratio
        horizontal = (not too_small) and track.is_horizontal()

        if track.fall_status == "None":
            if collapsed and horizontal:
                track.fall_status       = "Suspected"
                track.fall_status_since = now

        elif track.fall_status == "Suspected":
            # Same brief-occlusion tolerance as the Confirmed branch
            # below, via the same fall_recovery_since field (the two
            # states are mutually exclusive, so reusing it is safe).
            # Without this, one noisy not-horizontal sample mid-window,
            # for example furniture or another person briefly crossing
            # the box, would hard-reset progress to "None" and a genuine
            # fall could never accumulate confirm_secs of mostly
            # horizontal time.
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

    def _update_motion_fight_pair(self, id1, id2, t1, t2):
        """Motion/proximity backup signal: if two proximate tracks are
        BOTH moving fast and erratically relative to their own size,
        sustained for STATE_MOTION_FIGHT_CONFIRM_SECONDS, escalate both
        directly to Fighting regardless of what the violence classifier
        scored them.

        This exists because the classifier alone can under-score a real
        fight (motion blur, an off-training-distribution camera angle),
        and the state machine's escalation ladder otherwise has no path
        to Fighting without avg_score() clearing STATE_AGITATED_SCORE --
        a track stuck at, say, 0.39 never even reaches Agitated, so it
        never alerts and never triggers a recording. Requiring BOTH
        tracks to be moving fast (not just one) is deliberate: it is what
        separates "two people grappling" from "one person walking fast
        past someone standing still," which is common and not violence.

        Same Suspected-with-recovery-tolerance shape as _update_fall:
        a sustained-confirm timer that only resets after
        STATE_MOTION_FIGHT_RECOVER_SECONDS of low motion, so a single
        still frame mid-scuffle doesn't throw away confirm progress.

        Two guards below exist specifically to stop this from firing on
        a SINGLE real person who was never near anyone else -- reported
        in production as the live feed unblurring and a clip recording
        for someone just moving around alone. Both guards fail the same
        way a real duplicate-detection glitch looks: the person detector
        or tracker briefly represents one human as two overlapping boxes
        (a new track ID spawned on a momentary tracking hiccup, most
        likely exactly when someone is moving quickly -- which is also
        when this signal is looking for motion). Proximity distance
        alone (_check_proximity's job) cannot tell that apart from two
        real people standing close, since a duplicate box sits almost
        exactly where the original was.
        """
        if t1 is None or t2 is None:
            return
        if t1.state in (PersonState.FIGHTING, PersonState.ON_GROUND, PersonState.EMERGENCY):
            return
        if t2.state in (PersonState.FIGHTING, PersonState.ON_GROUND, PersonState.EMERGENCY):
            return

        cfg           = self.cfg
        intensity_thr = getattr(cfg, "STATE_MOTION_FIGHT_INTENSITY", 1.2)
        confirm_secs  = getattr(cfg, "STATE_MOTION_FIGHT_CONFIRM_SECONDS", 1.5)
        recover_secs  = getattr(cfg, "STATE_MOTION_FIGHT_RECOVER_SECONDS", 1.0)
        min_age       = getattr(cfg, "STATE_MOTION_FIGHT_MIN_TRACK_AGE_SECONDS", 1.0)
        max_iou       = getattr(cfg, "STATE_MOTION_FIGHT_MAX_IOU", 0.3)

        pair_key = tuple(sorted((id1, id2)))
        timer    = self._motion_fight_pairs.setdefault(pair_key, {"since": None, "recovery_since": None})

        now = time.time()

        # Guard 1: a track younger than min_age hasn't existed long
        # enough to be trusted as "a second real person" -- a genuine
        # duplicate-detection artifact is, by definition, a brand new
        # track id that just appeared. This does not reset the pair's
        # progress (a young track will age out of this guard within
        # min_age seconds if it's real), it just withholds judgment for
        # this frame the same way a too-small bbox withholds judgment in
        # _update_fall.
        too_young = (now - t1.first_seen < min_age) or (now - t2.first_seen < min_age)

        # Guard 2: heavily overlapping boxes are almost always one body
        # detected twice, not two people. Checked against each track's
        # latest bbox_history entry.
        overlapping = False
        if t1.bbox_history and t2.bbox_history:
            overlapping = _bbox_iou(t1.bbox_history[-1], t2.bbox_history[-1]) > max_iou

        both_fast = (not too_young and not overlapping
                     and t1.motion_intensity() >= intensity_thr
                     and t2.motion_intensity() >= intensity_thr)

        if both_fast:
            timer["recovery_since"] = None
            if timer["since"] is None:
                timer["since"] = now
            elif now - timer["since"] >= confirm_secs:
                t1.set_state(PersonState.FIGHTING)
                t2.set_state(PersonState.FIGHTING)
                t1.motion_confirmed_fight = True
                t2.motion_confirmed_fight = True
                timer["since"] = None
        else:
            if timer["since"] is not None:
                if timer["recovery_since"] is None:
                    timer["recovery_since"] = now
                elif now - timer["recovery_since"] > recover_secs:
                    timer["since"]          = None
                    timer["recovery_since"] = None

    def update(self, track_id, score, bbox, frame):
        """Update state for one tracked person."""
        cfg = self.cfg
        track = self._get_or_create(track_id)
        track.last_seen = time.time()
        track.update_bbox(bbox)
        track.update_score(score)
        track.update_height(bbox)
        track.update_position(bbox)
        self._update_fall(track)

        avg = track.avg_score()
        debounce_frames = getattr(cfg, "STATE_HORIZONTAL_DEBOUNCE_FRAMES", 3)

        # state transitions. Thresholds below are getattr-with-default so
        # a deployment can tune them via Config (see config.example.py's
        # "Violence state machine thresholds" section) without every
        # threshold needing an explicit value in every Config subclass.
        if track.state == PersonState.NORMAL:
            if avg >= getattr(cfg, "STATE_AGITATED_SCORE", 0.4):
                track.set_state(PersonState.AGITATED)
            # proximity handled separately in update_all

        elif track.state == PersonState.PROXIMATE:
            if avg >= getattr(cfg, "STATE_PROXIMATE_AGITATED_SCORE", 0.5):
                track.set_state(PersonState.AGITATED)
            elif (avg < getattr(cfg, "STATE_PROXIMATE_RECOVER_SCORE", 0.2)
                  and track.time_in_state() > getattr(cfg, "STATE_PROXIMATE_RECOVER_SECONDS", 5)):
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.AGITATED:
            if avg >= cfg.VIOLENCE_THRESHOLD:
                track.set_state(PersonState.FIGHTING)
            elif (avg < getattr(cfg, "STATE_AGITATED_RECOVER_SCORE", 0.3)
                  and track.time_in_state() > getattr(cfg, "STATE_AGITATED_RECOVER_SECONDS", 3)):
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.FIGHTING:
            if track.is_horizontal_sustained(debounce_frames):
                track.set_state(PersonState.ON_GROUND)
            elif (avg < getattr(cfg, "STATE_FIGHTING_RECOVER_SCORE", 0.3)
                  and track.time_in_state() > getattr(cfg, "STATE_FIGHTING_RECOVER_SECONDS", 5)):
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.ON_GROUND:
            if track.time_in_state() > getattr(cfg, "STATE_ON_GROUND_EMERGENCY_SECONDS", 30):
                track.set_state(PersonState.EMERGENCY)
            elif (track.is_vertical_sustained(debounce_frames)
                  and track.time_in_state() > getattr(cfg, "STATE_ON_GROUND_RECOVER_SECONDS", 3)):
                track.set_state(PersonState.NORMAL)

        elif track.state == PersonState.EMERGENCY:
            # Recovers back to Normal once the person has been sustained
            # off the ground (not a single noisy frame -- see
            # is_vertical_sustained) for STATE_EMERGENCY_RECOVER_SECONDS.
            # This used to have no recovery path at all ("stays until
            # manually reset"), but nothing in the codebase ever reset
            # it, so a track that reached Emergency was stuck there for
            # its entire remaining lifetime (until the 5s no-detections
            # cleanup dropped the track completely) even after the
            # person visibly got back up.
            if (track.is_vertical_sustained(debounce_frames)
                  and track.time_in_state() > getattr(cfg, "STATE_EMERGENCY_RECOVER_SECONDS", 5)):
                track.set_state(PersonState.NORMAL)

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
            self._update_motion_fight_pair(id1, id2, t1, t2)

        # Drop sustained-motion timers for pairs that are no longer
        # proximate this frame, so an old pair's partial progress can't
        # silently resume minutes later against unrelated people who
        # happen to reuse the same two track ids' relationship. This is
        # a hard reset rather than the brief-interruption tolerance
        # inside _update_motion_fight_pair itself, which is for a pair
        # that IS still proximate but had one noisy low-motion frame.
        current_pairs = {tuple(sorted(p)) for p in proximate_pairs}
        for stale_key in [k for k in self._motion_fight_pairs if k not in current_pairs]:
            del self._motion_fight_pairs[stale_key]

        # Moved here from update(): update() runs once per *detection*
        # in a frame (so N times for N people), but update_all() runs
        # exactly once per frame regardless of how many people are in
        # it (see pipeline.py's process_frame, which calls update_all
        # unconditionally after the detection loop). Cleanup only needs
        # to happen once per frame; running it once per detection did
        # the same O(tracks) sweep N times over for no benefit.
        self._cleanup_old_tracks()

    def get_states(self):
        """Returns dict of track_id -> state for dashboard display."""
        return {tid: t.state for tid, t in self.tracks.items()}

    def has_emergency(self):
        return any(t.state == PersonState.EMERGENCY for t in self.tracks.values())

    def has_fighting(self):
        return any(t.state == PersonState.FIGHTING for t in self.tracks.values())

    def has_fall(self):
        return any(t.fall_status == "Confirmed" for t in self.tracks.values())
