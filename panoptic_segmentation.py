# Install Detectron2 first (run these as separate Colab cells, not inside this file):
!rm -rf detectron2
!pip install -q ninja
!pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
# Detectron2 installation varies by Python / PyTorch / CUDA version; see the official docs.
import sys
import cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# -------------------------------
# Config (same as the YOLO script)
# -------------------------------
CONF_THRESH = 0.3
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FRAME_SIZE = (640, 480)
FRAME_SKIP = 5
MAX_SAMPLES = 30
SHOW_FRAMES = 4                # number of processed frames to display

LOW_MAX, MEDIUM_MAX = 5, 15
MIN_GREEN, MAX_GREEN = 10, 90
BASE_GREEN, SEC_PER_VEHICLE = 5, 2

# -------------------------------
# Upload video
# -------------------------------
if IN_COLAB:
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No video uploaded.")
    video_path = list(uploaded.keys())[0]
else:
    video_path = sys.argv[1] if len(sys.argv) > 1 else "project.mp4"

# -------------------------------
# Model
# -------------------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONF_THRESH
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
print(f"Running on: {cfg.MODEL.DEVICE}")

# -------------------------------
# Rules
# -------------------------------
def analyze_density(count):
    if count < LOW_MAX:
        return "Low"
    elif count < MEDIUM_MAX:
        return "Medium"
    return "High"

def calculate_signal_duration(count):
    return int(np.clip(BASE_GREEN + SEC_PER_VEHICLE * count, MIN_GREEN, MAX_GREEN))

def compute_metrics(vehicle_counts, all_confs):
    if len(vehicle_counts) < 2 or not all_confs:
        print("⚠️ Not enough data for metrics (need >=2 frames and >=1 detection).")
        return None
    mean_conf = float(np.mean(all_confs))
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
def main(video_path, max_samples=MAX_SAMPLES, frame_skip=FRAME_SKIP, save_csv=True):
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
        panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]

        class_counts = {n: 0 for n in VEHICLE_CLASSES.values()}
        confs = []
        for s in segments_info:
            if s.get("isthing", False) and s["category_id"] in VEHICLE_CLASSES:
                class_counts[VEHICLE_CLASSES[s["category_id"]]] += 1
                confs.append(float(s.get("score", 0)))

        count = len(confs)
        density = analyze_density(count)
        duration = calculate_signal_duration(count)

        print(f"Frame {frame_counter}: {count} vehicles {class_counts}, "
              f"Density={density}, Green={duration}s")

        if shown < SHOW_FRAMES:
            v = Visualizer(frame[:, :, ::-1], metadata, scale=1.0)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            plt.figure(figsize=(8, 6))
            plt.imshow(out.get_image())
            plt.axis("off"); plt.show()
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
        pd.DataFrame(log_data).to_csv("traffic_log.csv", index=False)
        print("✅ Results saved to traffic_log.csv")
        if IN_COLAB:
            files.download("traffic_log.csv")


main(video_path)