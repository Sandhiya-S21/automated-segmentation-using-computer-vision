# ============================================================
# TRAFFIC SEGMENTATION - SEMANTIC-STYLE VISUALIZATION
# YOLO11 segmentation: all vehicle masks merged into one
# combined mask (not a dedicated semantic segmentation model).
# Google Colab compatible; also runs locally:
#   python semantic_segmentation.py path/to/video.mp4
# ============================================================
# Colab install (run as a separate cell):
#   !pip install -q ultralytics opencv-python-headless pandas numpy matplotlib

import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import Counter
from ultralytics import YOLO
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# -------------------------------
# Config (same as the other scripts)
# -------------------------------
MODEL_NAME = "yolo11m-seg.pt"
CONF_THRESH = 0.3
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FRAME_SIZE = (640, 480)          # (width, height)
FRAME_SKIP = 5
MAX_SAMPLES = 30                 # processed frames, not raw frames
SHOW_FRAMES = 4

LOW_MAX, MEDIUM_MAX = 5, 15
MIN_GREEN, MAX_GREEN = 10, 90    # same limits as the other scripts
BASE_GREEN, SEC_PER_VEHICLE = 5, 2

MASK_COLOR = (0, 255, 0)         # BGR green
MASK_ALPHA = 0.45
OUTPUT_CSV = "traffic_ml_log.csv"

# -------------------------------
# Upload video
# -------------------------------
if IN_COLAB:
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No video uploaded.")
    video_path = next(iter(uploaded.keys()))
else:
    video_path = sys.argv[1] if len(sys.argv) > 1 else "project.mp4"

# -------------------------------
# Model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_NAME)
model.to(device)
print(f"Running on: {device}")

# -------------------------------
# Rules (same as the other scripts)
# -------------------------------
def analyze_density(count):
    if count < LOW_MAX:
        return "Low"
    elif count < MEDIUM_MAX:
        return "Medium"
    return "High"

def calculate_signal_duration(count):
    return int(np.clip(BASE_GREEN + SEC_PER_VEHICLE * count, MIN_GREEN, MAX_GREEN))

# -------------------------------
# Detection: masks only, drawn once as a union (no darker overlaps)
# -------------------------------
def detect_vehicles(frame):
    r = model.predict(source=frame, conf=CONF_THRESH,
                      classes=list(VEHICLE_CLASSES.keys()),
                      device=device, verbose=False)[0]

    union = np.zeros(frame.shape[:2], dtype=np.uint8)
    class_counts = Counter({n: 0 for n in VEHICLE_CLASSES.values()})
    confs_out = []

    if r.masks is not None and r.boxes is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for poly, cls, conf in zip(r.masks.xy, classes, confs):
            if cls not in VEHICLE_CLASSES or len(poly) < 3:
                continue
            cv2.fillPoly(union, [poly.astype(np.int32)], 1)
            class_counts[VEHICLE_CLASSES[cls]] += 1
            confs_out.append(float(conf))

    annotated = frame.copy()
    m = union.astype(bool)
    annotated[m] = (frame[m] * (1 - MASK_ALPHA) +
                    np.array(MASK_COLOR) * MASK_ALPHA).astype(np.uint8)
    return annotated, len(confs_out), confs_out, dict(class_counts)

def add_text(frame, frame_id, count, density, duration):
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Frame {frame_id}: {count} vehicles", (20, 35),
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Density={density}", (20, 70),
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Green={duration}s", (20, 105),
                font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

# -------------------------------
# Metrics (same as the other scripts)
# -------------------------------
def compute_metrics(vehicle_counts, all_confs):
    if len(vehicle_counts) < 2 or not all_confs:
        print("⚠️ Not enough data for metrics (need >=2 frames and >=1 detection).")
        return None
    mean_conf = float(np.mean(all_confs))                  # over every detection
    variance = float(np.var(vehicle_counts, ddof=1))
    mean_count = float(np.mean(vehicle_counts))
    cv = np.std(vehicle_counts, ddof=1) / mean_count if mean_count > 0 else 0.0
    stability = round(float(1 / (1 + cv)), 2)

    print(f"✅ Mean Detection Confidence (per detection): {mean_conf:.2f}")
    print(f"📉 Variance of Vehicle Counts: {variance:.2f}")
    print(f"🔄 Stability Score: {stability:.2f}")
    print(f"📊 Frames analysed: {len(vehicle_counts)}, total detections: {len(all_confs)}")
    return mean_conf, variance, stability

# -------------------------------
# Main loop
# -------------------------------
def main(video_path, max_samples=MAX_SAMPLES, frame_skip=FRAME_SKIP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_counter, shown = 0, 0
    log_data, vehicle_counts, all_confs = [], [], []

    while len(vehicle_counts) < max_samples:
        if frame_counter % frame_skip != 0:
            if not cap.grab():
                break
            frame_counter += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        annotated, count, confs, class_counts = detect_vehicles(frame)
        density = analyze_density(count)
        duration = calculate_signal_duration(count)

        print(f"Frame {frame_counter}: {count} vehicles {class_counts}, "
              f"Density={density}, Green={duration}s")

        if shown < SHOW_FRAMES:
            out = add_text(annotated, frame_counter, count, density, duration)
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
            shown += 1

        log_data.append({
            "Frame": frame_counter, "Vehicles": count,
            "MeanConfidence": float(np.mean(confs)) if confs else np.nan,
            "Density": density, "GreenDuration": duration,
            **{f"n_{k}": v for k, v in class_counts.items()},
        })
        vehicle_counts.append(count)
        all_confs.extend(confs)
        frame_counter += 1

    cap.release()
    compute_metrics(vehicle_counts, all_confs)

    if log_data:
        pd.DataFrame(log_data).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Results saved to {OUTPUT_CSV}")
        if IN_COLAB:
            files.download(OUTPUT_CSV)

main(video_path)