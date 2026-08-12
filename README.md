# 🚨 Fight Detection System

> AI-powered real-time violence detection for mental hospitals and elder care facilities.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![Flask](https://img.shields.io/badge/Flask-3.1-green) ![YOLO](https://img.shields.io/badge/YOLO-11n-purple)

---

## 📌 Overview

Fight Detection System monitors CCTV camera feeds in real-time, detects violent incidents using a trained deep learning model, and immediately alerts staff via email with an attached video clip. A live web dashboard shows all camera feeds, violence scores, and event history.

Built for internship deployment in institutional settings where rapid staff response to patient violence is critical.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎯 Violence Detection | EfficientNet-B0 classifier, Val F1: 0.8981, AUC: 0.9343 |
| 👤 Person Detection | YOLO11n with BoT-SORT tracking |
| 🧠 State Machine | 6 states per person: Normal → Proximate → Agitated → Fighting → On Ground → Emergency |
| 🌊 Motion Gating | Optical flow prevents false alerts on static scenes |
| 🎬 Clip Recording | 10s pre-event buffer + 15s post-event recording |
| 📧 Email Alerts | Instant notification with video clip attached |
| 📊 Dashboard | Live multi-camera feeds, violence scores, event log |
| 🔌 Multi-Camera | Add unlimited cameras via dashboard UI |
| 🔄 Swappable Model | Change model with one line in config |

---

## 🏗️ System Architecture

```
Camera (RTSP/Webcam)
        ↓
Person Detection (YOLO11n)
        ↓
Optical Flow Motion Check
        ↓
Violence Classifier (EfficientNet-B0)
        ↓
Per-Person State Machine
        ↓
Confirmation Gate (3s sustained + 0.9 threshold)
        ↓
    ┌───────────────────┐
    │  Clip Recording   │
    │  Email Alert      │
    │  Dashboard Log    │
    └───────────────────┘
```

---

## 📁 Project Structure

```
Fight-Detection/
├── src/
│   ├── main.py                    # Single entry point
│   ├── detection/
│   │   ├── pipeline.py            # Core inference pipeline
│   │   ├── detector.py            # YOLO person detector + tracker
│   │   ├── state_machine.py       # Per-person 6-state machine
│   │   ├── config.example.py      # Configuration template
│   │   └── config.py              # Your config (not in repo)
│   ├── dashboard/
│   │   ├── app.py                 # Flask web dashboard
│   │   └── templates/index.html   # Dashboard UI
│   └── notification/
│       └── notifier.py            # Email notification handler
├── outputs/
│   ├── clips/                     # Recorded alert clips
│   ├── logs/                      # Event log JSON
│   └── manifests/                 # Dataset CSVs
├── models/                        # Model weights (see Drive link)
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/naziaperwaiz-ai/Fight-Detection.git
cd Fight-Detection
```

### 2. Install Dependencies
```bash
pip install torch torchvision ultralytics opencv-python flask scikit-learn pandas requests
```

### 3. Download Model
Download `finetuned_model.pt` from Google Drive and place in `models/` folder:

🔗 **[Download Models from Google Drive](https://drive.google.com/drive/folders/1NG1qVZZ-JG_2WDhVJ5vqSXZbGCuk91Qq?usp=sharing)**

### 4. Configure
```bash
cp src/detection/config.example.py src/detection/config.py
```

Edit `src/detection/config.py`:
```python
MODEL_PATH         = "models/finetuned_model.pt"
CAMERA_SOURCE      = 0                    # 0 = webcam, or RTSP URL
EMAIL_SENDER       = "your@gmail.com"
EMAIL_APP_PASSWORD = "your-app-password"
EMAIL_RECIPIENTS   = ["staff@hospital.com"]
```

### 5. Run
```bash
py src/main.py
```

Open **http://localhost:5000** in your browser.

---

## 🧠 State Machine

Each tracked person transitions through 6 states independently:

```
Normal ──► Proximate ──► Agitated ──► Fighting ──► On Ground ──► Emergency
  ▲                          │                                       │
  └──────────────────────────┘                              (30s motionless)
```

| State | Trigger | Color |
|---|---|---|
| Normal | Default | 🟢 Green |
| Proximate | Two people close together | 🟡 Yellow |
| Agitated | Rapid movement, score > 0.4 | 🟠 Orange |
| Fighting | Score > 0.9 sustained 3s | 🔴 Red |
| On Ground | Bounding box wider than tall | 🟣 Purple |
| Emergency | On ground > 30 seconds | ⚠️ Alert |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Validation F1 | 0.8981 |
| ROC-AUC | 0.9343 |
| Validation Accuracy | 86.7% |

**Training Datasets:**
- RLVS — Real Life Violence Situations (2,000 clips)
- SCVD — Smart City CCTV Violence Detection (indoor surveillance)
- UCF-Crime — Fighting, Assault, Abuse, NormalVideos

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/finetuned_model.pt` | Swap model by changing this |
| `CAMERA_SOURCE` | `0` | Webcam index or RTSP URL |
| `VIOLENCE_THRESHOLD` | `0.90` | Alert trigger threshold |
| `CONFIRM_SECONDS` | `3` | Seconds of sustained detection before alert |
| `MOTION_THRESHOLD` | `1.5` | Optical flow threshold to skip static scenes |
| `BUFFER_SECONDS` | `10` | Pre-event recording buffer |
| `POST_EVENT_SECONDS` | `15` | Recording duration after alert |
| `COOLDOWN_SECONDS` | `120` | Minimum seconds between alerts per camera |

---

## 📧 Email Alerts

Two emails sent per event:
1. **Immediate alert** — camera ID, room, confidence score, timestamp
2. **Clip ready** — same info + video file attached

To set up Gmail app password: [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)

---

## 🛣️ Roadmap

- [ ] AWS Kinesis Video Streams integration
- [ ] SageMaker serverless inference endpoint
- [ ] S3 clip storage with presigned URLs
- [ ] WhatsApp/SMS notifications
- [ ] Thermal IR camera support
- [ ] YOLO26 upgrade when available in Ultralytics
- [ ] Staged indoor clip retraining for hospital domain

---

## 👥 Team

Built during summer internship for AI-powered institutional safety monitoring.