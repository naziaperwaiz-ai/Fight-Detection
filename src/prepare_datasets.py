# src/prepare_datasets.py
# Dataset preparation for fight/violence detection system.
# Reads RLVS, SCVD (sec_split), and UCF-Crime datasets.
# Extracts 1 fps frames, labels them violence/nonviolence,
# and writes train/val/test CSVs for the training pipeline.
#
# Run from: C:\Summer Internship\Fighting Cam\
# Command:  py src/prepare_datasets.py

import os
import cv2
import csv
import random
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR      = Path(".")
DATASETS_DIR  = BASE_DIR / "datasets"
OUTPUT_DIR    = BASE_DIR / "outputs" / "frames"   # extracted frames land here
MANIFEST_DIR  = BASE_DIR / "outputs" / "manifests" # CSV files land here

FPS_EXTRACT   = 1        # 1 frame per second
VAL_RATIO     = 0.15     # 15 % of training data becomes validation
TEST_RATIO    = 0.0      # UCF-Crime and SCVD already supply test splits; RLVS test comes from val
SEED          = 42

LABEL_VIOLENCE    = 1
LABEL_NONVIOLENCE = 0

# UCF-Crime categories to use (others like Robbery, Burglary are excluded per project spec)
UCF_VIOLENCE_CATS    = {"Fighting", "Assault", "Abuse"}
UCF_NONVIOLENCE_CATS = {"NormalVideos"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def video_extensions():
    return {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def collect_videos(folder: Path):
    """Return all video file paths under folder, recursively."""
    exts = video_extensions()
    videos = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in exts:
                videos.append(Path(root) / f)
    return sorted(videos)


def extract_frames(video_path: Path, out_dir: Path, fps: int = 1) -> list[str]:
    """
    Extract frames at `fps` frames-per-second from video_path.
    Saves as JPEG into out_dir.
    Returns list of saved frame paths (relative to BASE_DIR).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path.name}")
        return []

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0  # fallback for broken headers

    interval = max(1, round(source_fps / fps))   # take every Nth frame
    out_dir.mkdir(parents=True, exist_ok=True)

    saved   = []
    frame_i = 0
    saved_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % interval == 0:
            fname = out_dir / f"{video_path.stem}_f{saved_i:05d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved.append(str(fname.relative_to(BASE_DIR)))
            saved_i += 1
        frame_i += 1

    cap.release()
    return saved


def process_video_folder(
    video_folder: Path,
    label: int,
    split_tag: str,       # "train" | "test"
    dataset_tag: str,     # e.g. "rlvs", "scvd", "ucf"
    records: list
):
    """
    Iterate all videos in video_folder, extract frames, append rows to records.
    records row: (frame_path, label, split, dataset)
    """
    videos = collect_videos(video_folder)
    if not videos:
        print(f"  [WARN] No videos found in {video_folder}")
        return

    label_name = "violence" if label == LABEL_VIOLENCE else "nonviolence"
    out_base   = OUTPUT_DIR / dataset_tag / split_tag / label_name

    print(f"  {dataset_tag}/{split_tag}/{label_name} — {len(videos)} videos")
    for i, vp in enumerate(videos, 1):
        out_subdir = out_base / vp.stem
        frames = extract_frames(vp, out_subdir)
        for fp in frames:
            records.append((fp, label, split_tag, dataset_tag))
        if i % 50 == 0 or i == len(videos):
            print(f"    {i}/{len(videos)} done ({len(records)} total frames so far)")


# ─────────────────────────────────────────────
# DATASET PROCESSORS
# ─────────────────────────────────────────────

def process_rlvs(records_train, records_test):
    """
    RLVS has no built-in split. We treat everything as train material
    and carve val out later via ratio. All goes into records_train.
    """
    print("\n[RLVS] Real Life Violence Situations")
    root = DATASETS_DIR / "rlvs" / "Real Life Violence Dataset"

    violence_dir    = root / "Violence"
    nonviolence_dir = root / "NonViolence"

    if not violence_dir.exists():
        print(f"  [ERROR] Not found: {violence_dir}")
        return

    process_video_folder(violence_dir,    LABEL_VIOLENCE,    "train", "rlvs", records_train)
    process_video_folder(nonviolence_dir, LABEL_NONVIOLENCE, "train", "rlvs", records_train)


def process_scvd(records_train, records_test):
    """
    SCVD — use SCVD_converted_sec_split which already has Train/Test folders.
    Walk every subfolder inside Train and Test — each subfolder is a class.
    Map folder names to violence / nonviolence heuristically.
    """
    print("\n[SCVD] Smart City CCTV Violence Detection")
    root = DATASETS_DIR / "scvd" / "SCVD" / "SCVD_converted_sec_split"

    violence_keywords    = {"violence", "fight", "assault", "aggress", "abuse"}
    nonviolence_keywords = {"normal", "nonviolence", "no_violence", "noviolence", "nonfight"}

    for split_tag, split_folder, records in [
        ("train", root / "Train", records_train),
        ("test",  root / "Test",  records_test),
    ]:
        if not split_folder.exists():
            print(f"  [WARN] Missing: {split_folder}")
            continue

        subfolders = [d for d in split_folder.iterdir() if d.is_dir()]
        for sf in subfolders:
            name_lower = sf.name.lower()
            if any(k in name_lower for k in violence_keywords):
                label = LABEL_VIOLENCE
            elif any(k in name_lower for k in nonviolence_keywords):
                label = LABEL_NONVIOLENCE
            else:
                # Ambiguous name — peek at parent folder name, default violence-safe
                print(f"  [WARN] Ambiguous class folder: {sf.name} — defaulting to VIOLENCE. "
                      f"Rename if wrong.")
                label = LABEL_VIOLENCE

            process_video_folder(sf, label, split_tag, "scvd", records)


def process_ucf(records_train, records_test):
    """
    UCF-Crime is pre-extracted PNGs, no video extraction needed.
    Files named Abuse001_x264_0.png, _10.png, _20.png etc.
    We copy/reference every 10th file (i.e. every file, since they're already at 10-frame intervals).
    We sample 1-in-10 to approximate 1fps equivalent and avoid bloating the dataset.
    """
    print("\n[UCF-Crime] Surveillance Crime Dataset (subset) — pre-extracted frames")
    root = DATASETS_DIR / "ucf_crime"

    for split_tag, split_folder, records in [
        ("train", root / "Train", records_train),
        ("test",  root / "Test",  records_test),
    ]:
        if not split_folder.exists():
            print(f"  [WARN] Missing: {split_folder}")
            continue

        for cat_dir in sorted(split_folder.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_name = cat_dir.name
            if cat_name in UCF_VIOLENCE_CATS:
                label = LABEL_VIOLENCE
            elif cat_name in UCF_NONVIOLENCE_CATS:
                label = LABEL_NONVIOLENCE
            else:
                print(f"  [SKIP] Excluded UCF category: {cat_name}")
                continue

            # collect all PNGs, sample every 10th to approximate 1fps
            # (frames are at 10-frame intervals already, original videos ~25fps → every 10th ≈ 1fps)
            all_frames = sorted(cat_dir.glob("*.png"))
            MAX_NORMAL = 8000
            if cat_name == "NormalVideos" and split_tag == "train":
                random.seed(SEED)
                sampled = random.sample(all_frames, min(MAX_NORMAL, len(all_frames)))
            else:
                sampled = all_frames[::10]

            label_name = "violence" if label == LABEL_VIOLENCE else "nonviolence"
            print(f"  ucf/{split_tag}/{label_name} ({cat_name}) — {len(all_frames)} frames → {len(sampled)} sampled")

            for fp in sampled:
                records.append((str(fp.relative_to(BASE_DIR)), label, split_tag, "ucf"))


# ─────────────────────────────────────────────
# SPLIT AND WRITE CSV
# ─────────────────────────────────────────────

def carve_val(records_train: list, val_ratio: float, seed: int):
    """
    From records_train, split off val_ratio fraction as validation.
    Stratify by label so class balance is maintained.
    Returns (train_records, val_records).
    """
    random.seed(seed)
    violence    = [r for r in records_train if r[1] == LABEL_VIOLENCE]
    nonviolence = [r for r in records_train if r[1] == LABEL_NONVIOLENCE]

    def split_off(lst, ratio):
        random.shuffle(lst)
        cut = int(len(lst) * ratio)
        return lst[cut:], lst[:cut]   # (remaining_train, val)

    v_train,  v_val  = split_off(violence,    val_ratio)
    nv_train, nv_val = split_off(nonviolence, val_ratio)

    train = v_train + nv_train
    val   = v_val   + nv_val

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def write_csv(records: list, path: Path, split_override: str = None):
    """
    Write manifest CSV.
    Columns: frame_path, label, split, dataset
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "label", "split", "dataset"])
        for frame_path, label, split, dataset in records:
            writer.writerow([frame_path, label, split_override or split, dataset])
    print(f"  Wrote {len(records):>6} rows → {path}")


def print_stats(train, val, test):
    def counts(records):
        v  = sum(1 for r in records if r[1] == LABEL_VIOLENCE)
        nv = sum(1 for r in records if r[1] == LABEL_NONVIOLENCE)
        return v, nv, len(records)

    tv, tnv, tt = counts(train)
    vv, vnv, vt = counts(val)
    sv, snv, st = counts(test)

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│  FRAME COUNT SUMMARY                                │")
    print("├────────────┬──────────┬─────────────┬──────────────┤")
    print("│ Split      │ Violence │ NonViolence │ Total        │")
    print("├────────────┼──────────┼─────────────┼──────────────┤")
    print(f"│ Train      │ {tv:>8} │ {tnv:>11} │ {tt:>12} │")
    print(f"│ Val        │ {vv:>8} │ {vnv:>11} │ {vt:>12} │")
    print(f"│ Test       │ {sv:>8} │ {snv:>11} │ {st:>12} │")
    print(f"│ TOTAL      │ {tv+vv+sv:>8} │ {tnv+vnv+snv:>11} │ {tt+vt+st:>12} │")
    print("└────────────┴──────────┴─────────────┴──────────────┘")

    # class imbalance warning
    total_v  = tv + vv + sv
    total_nv = tnv + vnv + snv
    total    = total_v + total_nv
    if total > 0:
        ratio = total_v / total
        if ratio < 0.35 or ratio > 0.65:
            print(f"\n  [WARN] Class imbalance detected — violence is {ratio:.1%} of total.")
            print("         Consider oversampling violence or undersampling nonviolence during training.")
        else:
            print(f"\n  Class balance looks healthy — violence is {ratio:.1%} of total.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Fight Detection — Dataset Preparation")
    print("=" * 60)

    ensure_dirs(OUTPUT_DIR, MANIFEST_DIR)

    records_train = []   # all train-split records across datasets
    records_test  = []   # all test-split records across datasets

    process_rlvs(records_train, records_test)
    process_scvd(records_train, records_test)
    process_ucf(records_train, records_test)

    print("\n[SPLIT] Carving validation set from training data...")
    train_records, val_records = carve_val(records_train, VAL_RATIO, SEED)

    print("\n[CSV] Writing manifests...")
    write_csv(train_records, MANIFEST_DIR / "train.csv", split_override="train")
    write_csv(val_records,   MANIFEST_DIR / "val.csv",   split_override="val")
    write_csv(records_test,  MANIFEST_DIR / "test.csv",  split_override="test")

    # combined manifest — useful for Colab so one file covers everything
    all_records = train_records + val_records + records_test
    write_csv(all_records,   MANIFEST_DIR / "all.csv")

    print_stats(train_records, val_records, records_test)

    print("\n[DONE] Next step: upload outputs/frames/ and outputs/manifests/ to Google Drive,")
    print("       then open the Colab training notebook and mount Drive.")


if __name__ == "__main__":
    main()