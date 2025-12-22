# Driver Drowsiness Monitoring (YOLOv8 + MobileNetV2)

This repository packages the original Colab experiments into a reproducible, scriptable pipeline for detecting drowsiness in low-light conditions. It uses **YOLOv8** for detection and a fine-tuned **MobileNetV2** classifier for awake/drowsy classification on cropped regions, and now includes a YOLOv8 training entrypoint.

## Features
- Real-time video inference with YOLOv8 detection and optional MobileNetV2 classification.
- Lightweight preprocessing for low-light robustness (bilateral filter + CLAHE).
- Training script to fine-tune the MobileNetV2 head on your own labeled crops.
- Clear, configurable paths (no hard-coded Drive locations).

## Project structure
```
src/
  realtime_inference.py   # Run webcam/video inference with YOLOv8 (+ optional classifier)
  train_classifier.py     # Fine-tune MobileNetV2 on cropped awake/drowsy images
  train_detector.py       # Train YOLOv8 detector on your dataset
configs/
  dataset.example.yaml    # Example YOLO dataset config
requirements.txt          # Runtime dependencies
```

## Setup
1. (Recommended) Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train or download a YOLOv8 detection model (e.g., `best.pt`). Place it somewhere locally.

## Train the detector (YOLOv8)
1. Copy `configs/dataset.example.yaml` to a new file and update the dataset paths, class count, and names.
2. Run training:
```bash
python -m src.train_detector \
  --data configs/your_dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --img 640 \
  --batch 16 \
  --project runs/drowsiness \
  --name exp
```
- Detector metrics (mAP, losses) are written to `runs/.../results.csv` and TensorBoard logs; use `yolo val` on a held-out split for deeper evaluation.

## Preparing classifier data
Organize cropped eye/face images like this:
```
data/
  train/
    awake/
    drowsy/
  val/
    awake/
    drowsy/
```
Images should already be roughly aligned crops produced from your detection step.

## Train the classifier (MobileNetV2)
```bash
python -m src.train_classifier \
  --data-root ./data \
  --epochs 15 \
  --output ./artifacts/mobilenet_drowsiness.h5 \
  --report-dir ./artifacts/reports
```
- The script now emits `metrics.json` and `confusion_matrix.png` in the report directory for quick sanity checks.

## Run real-time inference
```bash
python -m src.realtime_inference \
  --yolo-weights /path/to/best.pt \
  --classifier-weights ./artifacts/mobilenet_drowsiness.h5 \
  --source 0
```
- `--source` can be a webcam index or a video file path.
- If `--classifier-weights` is omitted, the script will only draw YOLO detections.
- Use `--conf`/`--iou`/`--classes`/`--max-det` to control YOLO filtering, `--device` for GPU/CPU targeting, `--cls-threshold` for classifier decisions, and `--save-video` to export annotated output.

## Notes and tips
- Keep lighting consistent; CLAHE helps, but garbage input yields garbage output.
- Adjust `--conf` and `--iou` for detection strictness, and `--device` for GPU selection.
- For production, move to a GPU-capable environment and profile latency per frame.

## License
MIT
