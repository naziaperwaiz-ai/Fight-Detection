# src/main.py
import sys, os, threading, time
sys.path.insert(0, os.path.dirname(__file__))

from detection.config import Config
from detection.pipeline import CameraWorker
from dashboard.app import app, register_worker, load_cameras

# shared model lock for multi-camera
import torch
_model_lock = threading.Lock()

def start_worker(cam_config):
    cfg              = Config()
    cfg.CAMERA_ID    = cam_config["id"]
    cfg.ROOM_NAME    = cam_config["room"]
    cfg.CAMERA_SOURCE = cam_config["source"]
    cfg.VIOLENCE_THRESHOLD = float(cam_config.get("threshold", 0.7))

    worker = CameraWorker(cfg)
    register_worker(cfg.CAMERA_ID, worker)

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    print(f"[MAIN] Started worker for {cfg.CAMERA_ID} — {cfg.ROOM_NAME}")
    return worker


def run_dashboard():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    # start dashboard
    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()
    time.sleep(2)

    # start a worker for every camera in cameras.json
    cameras = load_cameras()
    workers = []
    for cam in cameras:
        w = start_worker(cam)
        workers.append(w)

    print(f"[MAIN] {len(workers)} camera(s) running. Press Ctrl+C to stop.")

    # keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MAIN] Shutting down.")