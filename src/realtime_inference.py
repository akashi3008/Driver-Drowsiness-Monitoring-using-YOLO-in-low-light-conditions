"""Real-time drowsiness monitoring using YOLOv8 + MobileNetV2.

This script wraps the experimental notebook code into a reproducible CLI.
It performs YOLO-based detection, crops regions, optionally classifies them
with a fine-tuned MobileNetV2, and overlays annotations on the video stream.

Usage:
    python -m src.realtime_inference \
        --yolo-weights /path/to/best.pt \
        --classifier-weights /path/to/mobilenet_weights.h5 \
        --source 0

Notes:
- The classifier is optional; if not provided, regions are only detected.
- The MobileNet head expects weights trained with `train_classifier.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array


def preprocess_image(frame: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering + CLAHE to stabilize low-light frames."""
    filtered = cv2.bilateralFilter(frame, 9, 75, 75)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def build_classifier(weights: Optional[Path]) -> Optional[Model]:
    """Load MobileNetV2 binary classifier; return None if weights missing."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    head = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=head)
    for layer in base.layers:
        layer.trainable = False
    if weights and weights.exists():
        model.load_weights(weights)
    else:
        print("[WARN] Classifier weights not provided; skipping classification.", file=sys.stderr)
        return None
    return model


def classify_region(region: np.ndarray, classifier: Model, threshold: float) -> str:
    """Return 'Drowsy' / 'Awake' based on classifier output."""
    if region.size == 0 or region.shape[0] == 0 or region.shape[1] == 0:
        return "Unknown"
    resized = cv2.resize(region, (224, 224))
    arr = img_to_array(resized)
    prepped = preprocess_input(np.expand_dims(arr, axis=0))
    pred = classifier.predict(prepped, verbose=0)[0][0]
    return "Drowsy" if pred > threshold else "Awake"


def build_augmentation():
    return A.Compose(
        [
            # Mild spatial/photometric jitter for robustness without destroying geometry.
            A.RandomResizedCrop(width=224, height=224, scale=(0.95, 1.0), ratio=(0.97, 1.03), p=0.8),
            A.HorizontalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.15),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def run_inference(args: argparse.Namespace) -> None:
    yolo_model = YOLO(args.yolo_weights)
    classifier = build_classifier(Path(args.classifier_weights) if args.classifier_weights else None)
    augment = build_augmentation()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(str(args.save_video), fourcc, fps, (width, height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = preprocess_image(frame)
            h, w, _ = processed.shape

            results = yolo_model(
                processed,
                verbose=False,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=args.max_det,
                classes=args.classes,
            )
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.int64)
                for x1, y1, x2, y2 in boxes:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1 = max(0, min(x1, w))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h))
                    y2 = max(0, min(y2, h))

                    region = processed[y1:y2, x1:x2]
                    state = "Detection"
                    if classifier is not None:
                        bbox_yolo = [
                            (x1 + x2) / 2 / w,
                            (y1 + y2) / 2 / h,
                            (x2 - x1) / w,
                            (y2 - y1) / h,
                        ]
                        augmented = augment(image=region, bboxes=[bbox_yolo], class_labels=[0])
                        region_aug = augmented["image"]
                        state = classify_region(region_aug, classifier, threshold=args.cls_threshold)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, state, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Drowsiness Monitor", frame)
            if writer is not None:
                writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", required=True, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument(
        "--classifier-weights",
        help="Optional MobileNetV2 classifier weights (.h5). If omitted, only detection is performed.",
    )
    parser.add_argument("--source", default=0, help="Video source (index or path)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        help="Class ids to filter for YOLO (default: all classes in the weights).",
    )
    parser.add_argument("--device", default=None, help="Computation device id, e.g. '0', '0,1', or 'cpu'")
    parser.add_argument("--cls-threshold", type=float, default=0.5, help="Classifier decision threshold")
    parser.add_argument(
        "--save-video",
        type=Path,
        help="Optional path to save annotated video (e.g., outputs/inference.mp4).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_inference(parse_args())
