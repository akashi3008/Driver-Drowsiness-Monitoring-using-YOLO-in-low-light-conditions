"""Fine-tune MobileNetV2 for drowsiness classification on cropped eye/face images.

Expected dataset layout:
    data/
      train/
        awake/
        drowsy/
      val/
        awake/
        drowsy/

Usage:
    python -m src.train_classifier \
        --data-root ./data \
        --epochs 15 \
        --output ./artifacts/mobilenet_drowsiness.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model() -> Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    head = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=head)
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_data(data_root: Path) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Dataset folders train/ and val/ not found under" f" {data_root!s}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
    ).map(lambda x, y: (preprocess_input(x), y))

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    ).map(lambda x, y: (preprocess_input(x), y))

    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)


def train(args: argparse.Namespace) -> None:
    model = build_model()
    train_ds, val_ds = load_data(Path(args.data_root))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    print("Training complete. Best weights saved to", args.output)
    for k, v in history.history.items():
        print(f"{k}: {v}")

    evaluate(model, val_ds, Path(args.report_dir), threshold=args.threshold)


def evaluate(model: Model, val_ds, report_dir: Path, threshold: float) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    y_true = []
    y_pred = []
    for batch_imgs, batch_labels in val_ds:
        preds = model.predict(batch_imgs, verbose=0).flatten()
        y_true.extend(batch_labels.numpy().flatten().tolist())
        y_pred.extend((preds > threshold).astype(int).tolist())

    import numpy as np
    import matplotlib.pyplot as plt
    import json

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    metrics = {
        "accuracy": acc,
        "precision_drowsy": precision,
        "recall_drowsy": recall,
        "f1_drowsy": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }

    with (report_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Awake", "Drowsy"])
    ax.set_yticklabels(["Awake", "Drowsy"])
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root folder containing train/ and val/ splits")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--output", default="mobilenet_drowsiness.h5", help="Where to store the trained weights")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive (drowsy)")
    parser.add_argument("--report-dir", default="artifacts/reports", help="Directory to save eval metrics/plots")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
