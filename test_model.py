import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from CNN_model import build_unet  # Only if you need to rebuild, otherwise skip

# Paths
test_img_dir = Path("dataset_test")
test_mask_dir = test_img_dir / "masks"

# Load trained model
model = tf.keras.models.load_model("Model_save_test.keras")

# Load test images & masks
def load_data(image_dir, mask_dir, img_size=(256,256)):
    image_paths = sorted([p for p in image_dir.glob("*.png")])
    mask_paths = sorted([p for p in mask_dir.glob("*_mask.png")])
    
    # Make sure the number of images and masks match
    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks!"
    
    images, masks = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = np.array(Image.open(img_path).resize(img_size)) / 255.0
        mask = np.array(Image.open(mask_path).resize(img_size)) / 255.0
        # Ensure mask is 0 or 1
        mask = np.expand_dims((mask > 0.5).astype(np.float32), axis=-1)
        images.append(img)
        masks.append(mask)
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

test_imgs, test_masks = load_data(test_img_dir, test_mask_dir)

# Predict
preds = model.predict(test_imgs, batch_size=2)
preds = (preds > 0.5).astype(np.float32)

# Dice Score function
def dice_score(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2 * intersection + smooth) / (union + smooth)

# Compute Dice Score
score = dice_score(test_masks, preds)
print("Dice Score on test set:", score)
