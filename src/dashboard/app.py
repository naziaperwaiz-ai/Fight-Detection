# src/dashboard/app.py
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from pathlib import Path
from datetime import datetime
import json, os, cv2, time

app = Flask(__name__)

_BASE = Path(__file__).parent.parent.parent
CLIPS_DIR    = _BASE / "outputs" / "clips"
LOGS_FILE    = _BASE / "outputs" / "logs" / "events.json"
CAMERAS_FILE = _BASE / "outputs" / "logs" / "cameras.json"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# registry of active CameraWorker instances
# key: cam_id, value: CameraWorker
_workers = {}

def register_worker(cam_id, worker):
    _workers[cam_id] = worker

def generate_frames(cam_id):
    while True:
        worker = _workers.get(cam_id)
        if worker:
            frame = worker.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        time.sleep(0.033)

def load_events():
    if LOGS_FILE.exists():
        return json.loads(LOGS_FILE.read_text())
    return []

def save_events(events):
    LOGS_FILE.write_text(json.dumps(events, indent=2))

def load_cameras():
    if CAMERAS_FILE.exists():
        return json.loads(CAMERAS_FILE.read_text())
    return [{"id":"CAM-01","room":"Ward A","source":"0","threshold":0.7,"active":True}]

def save_cameras(cameras):
    CAMERAS_FILE.write_text(json.dumps(cameras, indent=2))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    return Response(
        generate_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/score/<cam_id>")
def get_score(cam_id):
    worker = _workers.get(cam_id)
    score  = worker.score if worker else 0.0
    return jsonify({"score": score})

@app.route("/api/events")
def get_events():
    return jsonify(load_events()[-50:])

@app.route("/api/events/add", methods=["POST"])
def add_event():
    data   = request.json
    events = load_events()
    events.append({
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id":  data.get("camera_id","unknown"),
        "room":       data.get("room","unknown"),
        "event_type": data.get("event_type","unknown"),
        "confidence": data.get("confidence", 0.0),
        "clip_path":  data.get("clip_path",""),
    })
    save_events(events)
    return jsonify({"status": "ok"})

@app.route("/api/cameras")
def get_cameras():
    return jsonify(load_cameras())

@app.route("/api/cameras/add", methods=["POST"])
def add_camera():
    data    = request.json
    cameras = load_cameras()
    cameras.append({
        "id":        data.get("id", f"CAM-{len(cameras)+1:02d}"),
        "room":      data.get("room","Unknown Room"),
        "source":    data.get("source","0"),
        "threshold": float(data.get("threshold", 0.7)),
        "active":    True
    })
    save_cameras(cameras)
    return jsonify({"status": "ok"})

@app.route("/api/cameras/delete/<cam_id>", methods=["DELETE"])
def delete_camera(cam_id):
    cameras = [c for c in load_cameras() if c["id"] != cam_id]
    save_cameras(cameras)
    return jsonify({"status": "ok"})

@app.route("/api/cameras/update/<cam_id>", methods=["POST"])
def update_camera(cam_id):
    data    = request.json
    cameras = load_cameras()
    for c in cameras:
        if c["id"] == cam_id:
            c.update(data)
    save_cameras(cameras)
    return jsonify({"status": "ok"})

@app.route("/clips/<filename>")
def serve_clip(filename):
    return send_from_directory(str(CLIPS_DIR.absolute()), filename)

@app.route("/api/clips")
def get_clips():
    clips = sorted(CLIPS_DIR.glob("*.mp4"), reverse=True)
    return jsonify([c.name for c in clips[:20]])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)