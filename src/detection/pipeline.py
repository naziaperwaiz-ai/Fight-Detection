# src/detection/pipeline.py
import requests
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from collections import deque
from pathlib import Path
import time
import numpy as np
import threading

from detection.config import Config
from notification.notifier import Notifier


class ViolenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)


class CameraWorker:
    """
    Owns one camera. Runs detection in background thread.
    Dashboard reads latest annotated frame via get_frame().
    """
    def __init__(self, config: Config):
        self.cfg    = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        self.model = ViolenceClassifier()
        # weights_only=True restricts unpickling to tensors/plain data,
        # not arbitrary Python objects -- a .pt file is a pickle archive by
        # default, and loading one with weights_only=False (the old
        # default) lets a malicious model file execute arbitrary code the
        # moment it's loaded. Model uploads are admin-only in this app, but
        # "trusted admin" doesn't mean "attacker never gets a file past
        # them" -- fail loudly rather than silently falling back to the
        # unsafe path on older torch versions.
        try:
            state_dict = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            raise RuntimeError(
                "This torch version does not support weights_only=True. "
                "Upgrade torch (>=1.13) before loading model weights -- "
                "loading with weights_only=False is a code-execution risk."
            )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"[{config.CAMERA_ID}] Model loaded on {self.device}")

        from detection.detector import PersonDetector
        from detection.state_machine import StateMachine
        self.detector     = PersonDetector(device="cpu")
        self.state_machine = StateMachine(config)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.buffer        = deque(maxlen=config.BUFFER_SECONDS * config.FPS)
        self.violence_scores = deque(maxlen=config.CONFIRM_SECONDS * config.FPS)
        self.alert_active  = False
        self.last_alert_time = 0
        self.recording     = False
        self.record_frames = []
        self.record_start_time = None
        self.notifier      = Notifier(config)
        self._prev_gray    = None

        # shared frame for dashboard
        self._frame      = None
        self._frame_lock = threading.Lock()
        self.score       = 0.0

    def get_frame(self):
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def _set_frame(self, frame):
        with self._frame_lock:
            self._frame = frame.copy()

    def _motion_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._prev_gray = gray
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(magnitude.mean())

    def _predict(self, frame, detections):
        if self._motion_score(frame) < self.cfg.MOTION_THRESHOLD:
            return 0.0

        if not detections:
            return 0.0

        scores = []
        for track_id, bbox, crop in detections:
            with torch.no_grad():
                img  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inp  = self.transform(img).unsqueeze(0).to(self.device)
                out  = self.model(inp)
                score = torch.softmax(out, dim=1)[0][1].item()
                scores.append(score)
                self.state_machine.update(track_id, score, bbox, frame)

        return float(max(scores)) if scores else 0.0

    def run(self):
        source = int(self.cfg.CAMERA_SOURCE) if str(self.cfg.CAMERA_SOURCE).isdigit() else self.cfg.CAMERA_SOURCE
        cap    = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[{self.cfg.CAMERA_ID}] Cannot open camera: {self.cfg.CAMERA_SOURCE}")
            return

        print(f"[{self.cfg.CAMERA_ID}] Camera opened. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.buffer.append(frame.copy())
            detections = self.detector.detect(frame)
            score      = self._predict(frame, detections)
            self.state_machine.update_all(detections)
            frame      = self.detector.draw(frame, detections)
            self.score = score
            self.violence_scores.append(score)

            avg_score  = float(np.mean(self.violence_scores))
            confirmed  = (
                avg_score >= self.cfg.VIOLENCE_THRESHOLD and
                len(self.violence_scores) == self.violence_scores.maxlen
            )
            now        = time.time()
            cooldown_ok = (now - self.last_alert_time) > self.cfg.COOLDOWN_SECONDS

            if confirmed and cooldown_ok and not self.alert_active:
                self.alert_active      = True
                self.last_alert_time   = now
                self.recording         = True
                self.record_frames     = list(self.buffer)
                self.record_start_time = now
                print(f"[{self.cfg.CAMERA_ID}] ALERT! Score: {avg_score:.2f}")

                self.notifier.send_alert(
                    self.cfg.CAMERA_ID, self.cfg.ROOM_NAME,
                    "Violence Detected", avg_score, "Saving..."
                )
                # Snapshot each currently-tracked person's state/score so the
                # dashboard's incident detail view can show real "people
                # tracked" data instead of nothing -- this is genuinely
                # available in state_machine at the moment the alert fires.
                states_snapshot = [
                    {"track_id": tid, "state": track.state, "score": track.avg_score()}
                    for tid, track in self.state_machine.tracks.items()
                ]
                try:
                    requests.post(
                        f"{self.cfg.DASHBOARD_URL}/api/events/add",
                        json={
                            "camera_id":  self.cfg.CAMERA_ID,
                            "room":       self.cfg.ROOM_NAME,
                            "event_type": "Violence Detected",
                            "confidence": avg_score,
                            "clip_path":  "Saving...",
                            "states":     states_snapshot
                        },
                        headers={"X-Internal-Key": getattr(self.cfg, "INTERNAL_API_KEY", "")},
                        timeout=0.5
                    )
                except:
                    pass

            if self.recording:
                self.record_frames.append(frame.copy())
                if now - self.record_start_time >= self.cfg.POST_EVENT_SECONDS:
                    self._save_clip()
                    self.recording    = False
                    self.alert_active = False
                    self.record_frames = []

            # annotate frame
            color = (0, 0, 255) if avg_score >= self.cfg.VIOLENCE_THRESHOLD else (0, 255, 0)
            cv2.putText(frame, f"Violence: {avg_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if self.alert_active:
                cv2.putText(frame, "ALERT", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # draw state for each person
            states = self.state_machine.get_states()
            for tid, state in states.items():
                track = self.state_machine.tracks.get(tid)
                if track and track.bbox_history:
                    x1, y1, x2, y2 = track.bbox_history[-1]
                    state_color = {
                        "Normal":    (0, 255, 0),
                        "Proximate": (0, 255, 255),
                        "Agitated":  (0, 165, 255),
                        "Fighting":  (0, 0, 255),
                        "OnGround":  (255, 0, 255),
                        "Emergency": (255, 0, 0),
                    }.get(state, (255,255,255))
                    cv2.putText(frame, state, (x1, y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # emergency escalation
            if self.state_machine.has_emergency():
                now = time.time()
                if (now - self.last_alert_time) > self.cfg.COOLDOWN_SECONDS:
                    self.last_alert_time = now
                    self.notifier.send_alert(
                        self.cfg.CAMERA_ID, self.cfg.ROOM_NAME,
                        "EMERGENCY - Person Down 30s", 1.0, "Check camera immediately"
                    )
                    # This used to only send an email -- it never reached the
                    # dashboard's incident history, so Emergency events were
                    # invisible in the UI. Log it the same way as the other
                    # alert types so it shows up in Incident history/Analytics.
                    emergency_states = [
                        {"track_id": tid, "state": track.state, "score": track.avg_score()}
                        for tid, track in self.state_machine.tracks.items()
                    ]
                    try:
                        requests.post(
                            f"{self.cfg.DASHBOARD_URL}/api/events/add",
                            json={
                                "camera_id":  self.cfg.CAMERA_ID,
                                "room":       self.cfg.ROOM_NAME,
                                "event_type": "Emergency",
                                "confidence": 1.0,
                                "clip_path":  "",
                                "states":     emergency_states
                            },
                            headers={"X-Internal-Key": getattr(self.cfg, "INTERNAL_API_KEY", "")},
                            timeout=0.5
                        )
                    except:
                        pass

            self._set_frame(frame)

            cv2.imshow(f"Violence Detection - {self.cfg.CAMERA_ID}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _save_clip(self):
        if not self.record_frames:
            return
        Path(self.cfg.CLIPS_DIR).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path  = (Path(self.cfg.CLIPS_DIR) / f"alert_{self.cfg.CAMERA_ID}_{timestamp}.mp4").as_posix()
        h, w      = self.record_frames[0].shape[:2]
        writer    = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"),
            self.cfg.FPS, (w, h)
        )
        for f in self.record_frames:
            writer.write(f)
        writer.release()
        print(f"[{self.cfg.CAMERA_ID}] Clip saved: {out_path}")
        self.notifier.send_alert(
            self.cfg.CAMERA_ID, self.cfg.ROOM_NAME,
            "Clip Ready", 0.0, out_path
        )
        try:
            requests.post(
                f"{self.cfg.DASHBOARD_URL}/api/events/add",
                json={
                    "camera_id":  self.cfg.CAMERA_ID,
                    "room":       self.cfg.ROOM_NAME,
                    "event_type": "Clip Ready",
                    "confidence": 0.0,
                    "clip_path":  out_path
                },
                headers={"X-Internal-Key": getattr(self.cfg, "INTERNAL_API_KEY", "")},
                timeout=0.5
            )
        except:
            pass


class Pipeline:
    """Backwards compatible wrapper — single camera."""
    def __init__(self, config: Config):
        self.worker = CameraWorker(config)

    def run(self):
        self.worker.run()