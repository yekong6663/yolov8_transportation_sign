#!/usr/bin/env python3
"""Test the trained YOLOv8 traffic sign model on images or folders."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
BEST_PT = BASE_DIR / "runs" / "detect" / "traffic_signs" / "weights" / "best.pt"

CLASS_NAMES = {
    0: "Green_forward",
    1: "Green_left",
    2: "Green_right",
    3: "Parking",
    4: "Red_forward",
    5: "Red_left",
    6: "Red_right",
    7: "no_entry",
    8: "turn_left",
    9: "turn_right",
}

COLORS = {
    0: (0, 255, 0),    1: (0, 255, 128),   2: (128, 255, 0),
    3: (0, 128, 255),  4: (0, 0, 255),     5: (128, 0, 255),
    6: (255, 0, 255),  7: (255, 0, 0),     8: (255, 128, 0),
    9: (255, 255, 0),
}


def draw_boxes(img, results):
    """Draw YOLO detection results on image."""
    if results[0].boxes is None:
        return img

    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(cls_id, (0, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, '?')} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)

    return img


def test_image(model, image_path, output_dir=None, show=False):
    """Run inference on a single image."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return

    results = model(str(image_path))
    img = results[0].orig_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = draw_boxes(img, results)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{image_path.stem}_result{image_path.suffix}"
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

    if show:
        cv2.imshow("Traffic Sign Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_folder(model, folder, output_dir=None):
    """Run inference on all images in a folder."""
    folder = Path(folder)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not files:
        print(f"No images found in {folder}")
        return

    print(f"Testing {len(files)} images...")
    for f in sorted(files):
        print(f"  {f.name}")
        test_image(model, str(f), output_dir=output_dir, show=False)


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 Traffic Sign Model")
    parser.add_argument("source", nargs="?", help="Image file or folder to test")
    parser.add_argument("-o", "--output", default="test_results",
                        help="Output directory for result images")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show results in GUI window")
    args = parser.parse_args()

    if not BEST_PT.exists():
        print(f"Model not found: {BEST_PT}")
        print("Train first: python src/train.py")
        return

    print(f"Loading model: {BEST_PT}")
    model = YOLO(str(BEST_PT))

    if args.source:
        source = Path(args.source)
        if source.is_dir():
            test_folder(model, source, output_dir=args.output)
        else:
            test_image(model, str(source), output_dir=args.output, show=args.show)
    else:
        # Default: test on all validation images
        val_dir = BASE_DIR / "dataset" / "images" / "val"
        print(f"No source specified, testing on val set: {val_dir}")
        test_folder(model, val_dir, output_dir=args.output)


if __name__ == "__main__":
    main()
