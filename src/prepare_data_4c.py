#!/usr/bin/env python3
"""4-class data preparation: augmentation, YOLO label generation, train/val split."""

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # src/ → project root
IMAGE_DIR = BASE_DIR / "image"
DATASET_DIR = BASE_DIR / "dataset_4c"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

TARGET_PER_CLASS = {
    "Parking":    60,   # 1 orig → heavy aug
    "no_entry":   60,   # 58 orig → light aug
    "turn_left":  60,   # 8 orig → heavy aug
    "turn_right": 60,   # 26 orig → light aug
}

CLASS_NAMES = ["Parking", "no_entry", "turn_left", "turn_right"]
CLASS2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Augmentation helpers ──────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def random_rotate(img, angle_range=(-30, 30)):
    h, w = img.shape[:2]
    angle = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_brightness_contrast(img):
    pil = Image.fromarray(img)
    factor_b = random.uniform(0.4, 1.8)
    pil = ImageEnhance.Brightness(pil).enhance(factor_b)
    factor_c = random.uniform(0.5, 1.5)
    pil = ImageEnhance.Contrast(pil).enhance(factor_c)
    return np.array(pil)


def random_hue_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-20, 20)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.5, 1.5), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_blur(img):
    k = random.choice([3, 5])
    if k >= min(img.shape[0], img.shape[1]):
        k = 3
    return cv2.GaussianBlur(img, (k, k), 0)


def random_noise(img):
    noise = np.random.normal(0, random.randint(5, 20), img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def random_affine(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    dx = random.uniform(-0.08, 0.08) * w
    dy = random.uniform(-0.08, 0.08) * h
    pts2 = np.float32([
        [dx, dy],
        [w + random.uniform(-0.08, 0.08) * w, random.uniform(-0.08, 0.08) * h],
        [random.uniform(-0.08, 0.08) * w, h + random.uniform(-0.08, 0.08) * h],
    ])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_augmentations(img, strength="heavy"):
    img = img.copy()

    if strength == "heavy":
        img = random_rotate(img, (-40, 40))
    else:
        img = random_rotate(img, (-20, 20))

    img = random_brightness_contrast(img)
    img = random_hue_saturation(img)

    if strength == "heavy":
        if random.random() < 0.6:
            img = random_blur(img)
        if random.random() < 0.5:
            img = random_noise(img)
        if random.random() < 0.5:
            img = random_affine(img)
    else:
        if random.random() < 0.3:
            img = random_blur(img)
        if random.random() < 0.3:
            img = random_noise(img)
        if random.random() < 0.3:
            img = random_affine(img)

    return img


# ── YOLO label helpers ────────────────────────────────────────────────────

def generate_yolo_label(class_id):
    return f"{class_id} 0.5 0.5 1.0 1.0"


# ── Main pipeline ─────────────────────────────────────────────────────────

def prepare_dataset():
    print("=" * 60)
    print("Preparing YOLOv8 4-Class Traffic Sign Dataset")
    print("=" * 60)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        src_dir = IMAGE_DIR / class_name
        if not src_dir.exists():
            print(f"  WARNING: directory not found: {src_dir}")
            continue

        orig_files = sorted([f for f in os.listdir(src_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        if not orig_files:
            print(f"  WARNING: no images in {class_name}")
            continue

        class_id = CLASS2ID[class_name]
        target = TARGET_PER_CLASS[class_name]
        n_orig = len(orig_files)

        if n_orig <= 2:
            strength = "heavy"
        elif n_orig <= 10:
            strength = "heavy"
        else:
            strength = "light"

        augs_per_img = max(1, target // n_orig)
        remainder = target - (augs_per_img * n_orig)

        print(f"\n  {class_name}: {n_orig} originals → target {target} "
              f"({augs_per_img} augs/img, strength={strength})")

        all_images = []

        for fname in orig_files:
            img_path = src_dir / fname
            img = load_image(img_path)
            if img is None:
                print(f"    WARNING: cannot load {img_path}")
                continue

            stem = Path(fname).stem
            all_images.append((img.copy(), f"{class_name}_{stem}_orig"))

            for j in range(augs_per_img - 1):
                aug_img = apply_augmentations(img, strength)
                all_images.append((aug_img, f"{class_name}_{stem}_aug{j:03d}"))

        for j in range(remainder):
            idx = j % n_orig
            fname = orig_files[idx]
            img = load_image(src_dir / fname)
            if img is not None:
                aug_img = apply_augmentations(img, strength)
                all_images.append((aug_img, f"{class_name}_{Path(fname).stem}_extra{j:03d}"))

        random.shuffle(all_images)
        n_train = int(len(all_images) * TRAIN_RATIO)
        train_set = all_images[:n_train]
        val_set = all_images[n_train:]

        print(f"    train={len(train_set)}, val={len(val_set)}")

        for split_name, img_list in [("train", train_set), ("val", val_set)]:
            img_dir = DATASET_DIR / "images" / split_name
            lbl_dir = DATASET_DIR / "labels" / split_name

            for img_arr, stem in img_list:
                out_name = f"{stem}.jpg"
                save_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_dir / out_name), save_img)
                label = generate_yolo_label(class_id)
                (lbl_dir / f"{stem}.txt").write_text(label + "\n")

    # ── Write data.yaml ───────────────────────────────────────────────────
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_content = f"""# YOLOv8 4-Class Traffic Sign Dataset
path: {DATASET_DIR}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path.write_text(yaml_content)

    # ── Summary ───────────────────────────────────────────────────────────
    n_train_imgs = len(list((DATASET_DIR / "images" / "train").glob("*.jpg")))
    n_val_imgs = len(list((DATASET_DIR / "images" / "val").glob("*.jpg")))
    n_train_lbls = len(list((DATASET_DIR / "labels" / "train").glob("*.txt")))
    n_val_lbls = len(list((DATASET_DIR / "labels" / "val").glob("*.txt")))

    print(f"\n{'=' * 60}")
    print(f"Dataset ready: {n_train_imgs} train + {n_val_imgs} val images")
    print(f"Labels: {n_train_lbls} train + {n_val_lbls} val")
    print(f"Config: {yaml_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    prepare_dataset()
