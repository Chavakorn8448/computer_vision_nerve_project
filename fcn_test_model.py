import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import json
import tensorflow as tf

# ==========================
# Config
# ==========================
TEST_DIR = Path("dataset_test")   # contains *.png and matching *.json
MODEL_PATH = "final_fcn8s.keras"  # or "nerve_unet_model.keras"
IMG_SIZE = (256, 256)


# ==========================
# Mask utils
# ==========================
def json_to_mask(json_path: Path, out_size=(256, 256)):
    """Convert LabelMe-style JSON polygons into a binary mask."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W, H = data.get("imageWidth"), data.get("imageHeight")
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        pts = shape.get("points", [])
        if not pts:
            continue
        draw.polygon([tuple(p) for p in pts], outline=1, fill=1)

    mask = mask_img.resize(out_size, resample=Image.NEAREST)
    mask = np.array(mask, dtype=np.float32)
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


# ==========================
# Data loader
# ==========================
def load_test_data(data_dir: Path, img_size=(256, 256)):
    images, masks = [], []
    pngs = sorted(data_dir.glob("*.png"))

    for img_path in pngs:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"⚠️ Skipping {img_path.name}, no JSON found")
            continue

        # Image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_size[1], img_size[0]))
        img = np.array(img, dtype=np.float32) / 255.0

        # Mask
        mask = json_to_mask(json_path, out_size=img_size)

        images.append(img)
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


# ==========================
# Metrics
# ==========================
def dice_score(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2 * intersection + smooth) / (union + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# ==========================
# Main
# ==========================
def main():
    print("Loading test set...")
    X_test, y_test = load_test_data(TEST_DIR, img_size=IMG_SIZE)
    print(f"Test set: {X_test.shape[0]} samples")

    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coefficient": lambda y_true, y_pred: dice_score(y_true.numpy(), y_pred.numpy())
        },
        compile=False
    )

    print("Predicting...")
    preds = model.predict(X_test, batch_size=2, verbose=1)
    preds_bin = (preds > 0.5).astype(np.float32)

    print("Computing metrics...")
    dice = dice_score(y_test, preds_bin)
    iou = iou_score(y_test, preds_bin)

    print(f"Dice Score on test set: {dice:.4f}")
    print(f"IoU Score  on test set: {iou:.4f}")

if __name__ == "__main__":
    main()
