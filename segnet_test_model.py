"""
Test/evaluate SegNet on LabelMe-style PNG+JSON dataset.
- Expects: dataset_test/*.png with matching .json (same stem)
- Loads: segnet_model.keras (no args needed)
- Reports: Dice and IoU
"""
import os
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# =========================
# Config
# =========================
DATA_DIR = Path("dataset_test")      # contains *.png and matching *.json
IMG_SIZE = (256, 256)                 # resize target (H, W)
MODEL_PATH = "segnet_model.keras"

# =========================
# Mask utils (LabelMe-style)
# =========================

def labelme_json_to_mask(json_path: Path, out_size) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = data.get("imageHeight", out_size[0])
    W = data.get("imageWidth", out_size[1])

    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if not pts:
            continue
        draw.polygon([tuple(p) for p in pts], outline=1, fill=1)

    mask = mask_img.resize((out_size[1], out_size[0]), resample=Image.NEAREST)
    mask = np.array(mask, dtype=np.float32)
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
    return mask


# =========================
# Data loading
# =========================

def load_dataset(data_dir: Path, img_size=(256, 256)):
    pngs = sorted([p for p in data_dir.glob("*.png")])
    images, masks, names = [], [], []

    for img_path in pngs:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"⚠️  Skipping {img_path.name}: missing {json_path.name}")
            continue

        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_size[1], img_size[0]))
        img = np.asarray(img, dtype=np.float32) / 255.0

        mask = labelme_json_to_mask(json_path, out_size=img_size)

        images.append(img)
        masks.append(mask)
        names.append(img_path.name)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    return images, masks, names


# =========================
# Metrics
# =========================

def dice_np(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2 * intersection + smooth) / (union + smooth)


def iou_np(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# =========================
# Main
# =========================

def main():
    print("Loading test dataset...")
    X_test, y_test, names = load_dataset(DATA_DIR, img_size=IMG_SIZE)
    print(f"Test samples: {X_test.shape[0]}")

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Predicting...")
    preds = model.predict(X_test, batch_size=2, verbose=1)
    preds_bin = (preds > 0.5).astype(np.float32)

    print("Computing metrics...")
    dice = dice_np(y_test, preds_bin)
    iou = iou_np(y_test, preds_bin)
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score : {iou:.4f}")

    # Optional per-image metrics (uncomment to print)
    # for i, name in enumerate(names):
    #     d = dice_np(y_test[i], preds_bin[i])
    #     j = iou_np(y_test[i], preds_bin[i])
    #     print(f"{name}: Dice={d:.4f}, IoU={j:.4f}")

if __name__ == "__main__":
    main()
