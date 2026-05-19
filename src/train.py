#!/usr/bin/env python3
"""Train YOLOv8 on the traffic sign dataset."""

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent  # src/ → project root
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"


def main():
    # Load pretrained YOLOv8 nano model (best speed/accuracy tradeoff for small datasets)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=320,            # smaller size for CPU
        batch=8,              # smaller batch for CPU
        name="traffic_signs",
        patience=20,          # early stopping
        save=True,
        save_period=10,
        val=True,
        plots=True,
        device="cpu",
        workers=2,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,             # initial learning rate
        lrf=0.01,             # final lr = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,          # cosine LR schedule
        close_mosaic=10,      # disable mosaic augmentation in last 10 epochs
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # Validate
    metrics = model.val()
    print(f"\nValidation results: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

    # Export to ONNX for deployment
    model.export(format="onnx")

    print(f"\nBest model saved at: {model.trainer.save_dir / 'weights' / 'best.pt'}")
    print("Done!")


if __name__ == "__main__":
    main()
