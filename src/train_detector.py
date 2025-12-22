"""Train a YOLOv8 detector for drowsiness/driver monitoring datasets.

Example:
    python -m src.train_detector \
      --data configs/dataset.example.yaml \
      --model yolov8n.pt \
      --epochs 50 \
      --img 640 \
      --batch 16 \
      --project runs/drowsiness \
      --name exp

The script is intentionally light on defaults and exposes common knobs so it can
run in local, Colab, or server environments without hard-coded paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML (train/val paths, nc, names)")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img", type=int, default=640, help="Training image size (square)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device id (e.g., '0', '0,1', or 'cpu')")
    parser.add_argument("--project", default="runs/detect", help="Project directory for outputs")
    parser.add_argument("--name", default="train", help="Run name inside the project directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint in project/name")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Optimizer weight decay")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze (0 = none)")
    return parser.parse_args(argv)


def train_detector(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        freeze=args.freeze,
        exist_ok=True,
    )


if __name__ == "__main__":
    train_detector(parse_args())
