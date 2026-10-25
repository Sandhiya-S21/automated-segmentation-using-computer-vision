# Comparison of Traffic Segmentation Approaches

## 1. Overview

This project compares three computer vision approaches for vehicle segmentation and traffic analysis:

1. YOLO11-based combined vehicle-mask visualization (semantic-style).
2. YOLO11-based instance segmentation.
3. Detectron2 Panoptic FPN-based panoptic segmentation.

All approaches analyse traffic video frames, count selected vehicle classes, estimate traffic density, and calculate an illustrative green signal duration.

## 2. Experimental Configuration

* Input: Traffic video
* Frames analysed per run: 30
* Frame sampling interval: Every 5 frames
* Input frame size: 640 × 480
* Vehicle classes: Car, motorcycle, bus, and truck
* Detection confidence threshold: 0.3
* Density thresholds: Low (<5), Medium (5–14), High (15 or more)
* Signal timing: A predefined vehicle-count-based rule

## 3. Results

| Metric                    | Semantic-style | Instance | Panoptic |
| ------------------------- | -------------: | -------: | -------: |
| Frames analysed           |             30 |       30 |       30 |
| Total detections          |            397 |      397 |      774 |
| Mean detection confidence |           0.67 |     0.67 |     0.68 |
| Vehicle-count variance    |           0.87 |     0.87 |     3.89 |
| Count stability score     |           0.93 |     0.93 |     0.93 |

All three runs use the same video and the same sampled frames (frames 0–145, every 5th frame). Values are model outputs, not ground-truth accuracy measurements.

## 4. Interpretation

### YOLO11 Semantic-Style Visualization

The implementation combines the vehicle instance masks produced by YOLO11 into a single mask visualization. It is useful for displaying the overall regions occupied by detected vehicles, but it is not a dedicated semantic segmentation model.

### YOLO11 Instance Segmentation

The implementation visualizes individual vehicle instances using separate mask colours. It provides a view of the individual detected vehicles and their classes.

The semantic-style and instance implementations use the same YOLO11 segmentation model and detection settings. Their total detection counts and confidence metrics can therefore be identical.

### Detectron2 Panoptic Segmentation

The panoptic implementation uses Detectron2 Panoptic FPN with a ResNet-50 backbone. It produces panoptic segmentation predictions and uses vehicle segments to calculate frame-level counts.

The reported total of 774 detections is higher than the 397 detections reported by the YOLO11 runs. This difference does not establish that either model is more accurate.

## 5. Traffic Density and Signal Duration

Traffic density is classified using predefined vehicle-count thresholds:

* Low: Fewer than 5 vehicles
* Medium: 5–14 vehicles
* High: 15 or more vehicles

The estimated green signal duration follows this rule:

Green Duration = 5 + (2 × Number of Detected Vehicles)

The result is clipped to a minimum of 10 seconds and a maximum of 90 seconds.

This is a demonstration heuristic, not a learned or validated traffic signal optimization model.

## 6. Metric Limitations

* Mean detection confidence measures model confidence, not correctness.
* Vehicle-count variance measures how much counts change across sampled frames.
* The stability score describes count consistency under the implemented formula; it does not measure segmentation accuracy.
* A reliable accuracy comparison requires labelled ground-truth data and appropriate metrics, such as precision, recall, and mask Intersection over Union (IoU).
* All models should be evaluated on the same input video and sampled frames for a fair comparison.

## 7. Conclusion

The project demonstrates three approaches to vehicle segmentation and traffic analysis. YOLO11 supports combined-mask visualization and instance segmentation, while Detectron2 Panoptic FPN provides a separate panoptic segmentation approach.

The results illustrate differences in detected vehicle counts and confidence values. Further evaluation with ground-truth annotations is required before drawing conclusions about comparative accuracy or suitability for real-world traffic control.
