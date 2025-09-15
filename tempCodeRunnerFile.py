import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from CNN_model import build_unet

# Paths
test_img_dir = Path("dataset_test")
test_mask_dir = test_img_dir / "masks"

# Load model
model = tf.keras.models.load_model("nerve_unet_model.keras")

# Load test images & masks
def load_data(image_dir, mask_dir, img_size=(256,256)):
    images, masks = [], []
    image_paths = sorted(image_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).resize(img_size)) / 255.0
        mask = np.expand_dims(np.array(Image.open(mask_path).resize(img_size)) / 255.0, axis=-1)
        images.append(img)
        masks.append(mask)
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

test_imgs, test_masks = load_data(test_img_dir, test_mask_dir)

# Predict
preds = model.predict(test_imgs)
preds = (preds > 0.5).astype(np.float32)

# Compute Dice Score
intersection = np.sum(preds * test_masks)
union = np.sum(preds) + np.sum(test_masks)
dice_score = 2 * intersection / union
print("Dice Score on test set:", dice_score)
