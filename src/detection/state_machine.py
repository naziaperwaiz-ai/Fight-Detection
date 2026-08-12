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

    def update(self, track_id, score, bbox, frame):
        """Update state for one tracked person."""
        track = self._get_or_create(track_id)
        track.last_seen = time.time()
        track.update_bbox(bbox)
        track.update_score(score)

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