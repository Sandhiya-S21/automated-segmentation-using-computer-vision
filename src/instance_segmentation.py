# Install dependencies (Colab):
#  !pip install ultralytics opencv-python-headless matplotlib pandas

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "yolo11m-seg.pt"      # "yolo11n-seg.pt" is faster but less accurate
CONF_THRESH = 0.3
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FRAME_SIZE = (640, 480)            # (width, height)
FRAME_SKIP = 5                     # process every Nth frame
MAX_SAMPLES = 30                   # processed frames (not raw frames)
SHOW_FRAMES = 4                    # plot only the first N processed frames

# "instance": every vehicle gets its own colour
# "semantic": every vehicle CLASS gets one colour (car=green, bus=blue, ...)
# Counting always uses the instance masks, so counts are identical in both modes.
VIEW_MODE = "instance"

# Same density / signal rules as the other script, so results are comparable
LOW_MAX, MEDIUM_MAX = 5, 15        # <5 Low, <15 Medium, else High
MIN_GREEN, MAX_GREEN = 10, 90
BASE_GREEN, SEC_PER_VEHICLE = 5, 2

CLASS_COLORS = {"car": (0, 255, 0), "motorcycle": (0, 165, 255),
                "bus": (255, 128, 0), "truck": (0, 0, 255)}          # BGR
_rng = np.random.default_rng(42)                                       # fixed -> no flicker
INSTANCE_PALETTE = _rng.integers(60, 255, size=(64, 3)).tolist()


# -------------------------------
# Detection
# -------------------------------
def detect_vehicles(frame, model):
    """Returns (annotated_frame, count, confidences, per_class_counts)."""
    r = model(frame, conf=CONF_THRESH, verbose=False)[0]
    overlay = frame.copy()
    confs_out = []
    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}

    if r.masks is not None and r.boxes is not None:
        polygons = r.masks.xy                      # pixel coords of the input frame
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        idx = 0
        for poly, conf, cls in zip(polygons, confs, classes):
            if cls not in VEHICLE_CLASSES or len(poly) < 3:
                continue
            name = VEHICLE_CLASSES[cls]
            color = CLASS_COLORS[name] if VIEW_MODE == "semantic" \
                else tuple(INSTANCE_PALETTE[idx % len(INSTANCE_PALETTE)])
            pts = poly.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 1)
            confs_out.append(float(conf))
            class_counts[name] += 1
            idx += 1

    # Only mask pixels differ between overlay and frame, so the background is untouched
    blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return blended, len(confs_out), confs_out, class_counts


def analyze_density(count):
    if count < LOW_MAX:
        return "Low"
    elif count < MEDIUM_MAX:
        return "Medium"
    return "High"


def calculate_signal_duration(count):
    return int(np.clip(BASE_GREEN + SEC_PER_VEHICLE * count, MIN_GREEN, MAX_GREEN))


def show_frame(frame, count, density, duration, frame_id):
    frame = frame.copy()
    cv2.putText(frame, f"Frame {frame_id}: {count} vehicles", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Density={density}, Green={duration}s", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    plt.axis("off")
    plt.show()


# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(vehicle_counts, all_confs):
    if len(vehicle_counts) < 2 or not all_confs:
        print("⚠️ Not enough data for metrics (need >=2 frames and >=1 detection).")
        return None

    mean_conf = float(np.mean(all_confs))                 # over every detection
    variance = float(np.var(vehicle_counts, ddof=1))      # sample variance
    mean_count = float(np.mean(vehicle_counts))
    cv = np.std(vehicle_counts, ddof=1) / mean_count if mean_count > 0 else 0.0
    stability = round(float(1 / (1 + cv)), 2)             # always in (0, 1]

    print(f"✅ Mean Detection Confidence (per detection): {mean_conf:.2f}")
    print(f"📉 Variance of Vehicle Counts: {variance:.2f}")
    print(f"🔄 Stability Score: {stability:.2f}")
    print(f"📊 Frames analysed: {len(vehicle_counts)}, total detections: {len(all_confs)}")
    return mean_conf, variance, stability


# -------------------------------
# Main loop
# -------------------------------
def main(video_path, max_samples=MAX_SAMPLES, frame_skip=FRAME_SKIP, save_csv=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    model = YOLO(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


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
        annotated, count, confs, class_counts = detect_vehicles(frame, model)
        density = analyze_density(count)
        duration = calculate_signal_duration(count)

        print(f"Frame {frame_counter}: {count} vehicles {class_counts}, "
              f"Density={density}, Green={duration}s")

        if shown < SHOW_FRAMES:
            show_frame(annotated, count, density, duration, frame_counter)
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

    if save_csv and log_data:
        pd.DataFrame(log_data).to_csv("instance_accuracy_log.csv", index=False)
        print("✅ Results saved to instance_accuracy_log.csv")
        if IN_COLAB:
            files.download("instance_accuracy_log.csv")


# -------------------------------
# Run
# -------------------------------
if IN_COLAB:
    uploaded = files.upload()
    video_path = list(uploaded.keys())[0]
else:
    video_path = sys.argv[1] if len(sys.argv) > 1 else "project.mp4"

main(video_path)
