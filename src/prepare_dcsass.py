# src/prepare_dcsass.py
# Extracts normal (label=0) frames from DCSASS dataset.
# Uses Labels CSVs to identify which frames are normal.
# Saves to datasets/dcsass_normal/ for fine-tuning.

import cv2
import pandas as pd
from pathlib import Path

DCSASS_DIR  = Path("datasets/dcsass/DCSASS Dataset")
LABELS_DIR  = DCSASS_DIR / "Labels"
OUTPUT_DIR  = Path("outputs/frames/dcsass/train/nonviolence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_EVERY = 5   # take every 5th normal frame to avoid too many similar frames

total_saved = 0
manifest_rows = []

for csv_file in sorted(LABELS_DIR.glob("*.csv")):
    category = csv_file.stem
    video_dir = DCSASS_DIR / category

    if not video_dir.exists():
        print(f"[SKIP] No folder for {category}")
        continue

    df = pd.read_csv(csv_file, header=None)
    df.columns = ["frame_name", "category", "label"]

    # get only normal frames
    normal_frames = df[df["label"] == 0]["frame_name"].tolist()
    print(f"{category}: {len(normal_frames)} normal frames across all videos")

    # group by video
    videos = {}
    for fname in normal_frames:
        # frame name is like Fighting002_x264_5
        # video is Fighting002_x264
        parts = fname.rsplit("_", 1)
        video_stem = parts[0]
        frame_idx  = int(parts[1])
        videos.setdefault(video_stem, []).append(frame_idx)

    for video_stem, frame_indices in videos.items():
        video_path = video_dir / f"{video_stem}.mp4"
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {video_path.name}")
            continue

        frame_set = set(frame_indices[::SAMPLE_EVERY])
        frame_i   = 0
        saved_i   = 0

        out_dir = OUTPUT_DIR / video_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i in frame_set:
                fname = out_dir / f"{video_stem}_n{frame_i:05d}.jpg"
                cv2.imwrite(str(fname), frame)
                rel = str(fname.relative_to(Path(".")))
                manifest_rows.append((rel, 0, "train", "dcsass"))
                saved_i += 1
                total_saved += 1
            frame_i += 1

        cap.release()

print(f"\nTotal normal frames saved: {total_saved}")

# append to existing train.csv
import csv
manifest_path = Path("outputs/manifests/train.csv")
with open(manifest_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in manifest_rows:
        writer.writerow(row)

print(f"Appended {len(manifest_rows)} rows to train.csv")