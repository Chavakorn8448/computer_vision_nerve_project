import os
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
DATA_DIR = Path("dataset_train")      # contains *.png and matching *.json
IMG_SIZE = (256, 256)                 # resize target (H, W)
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42


# =========================
# Mask utils (LabelMe-style)
# =========================
def labelme_json_to_mask(json_path: Path, out_size) -> np.ndarray:
    """
    Build a binary mask from a LabelMe-style JSON file with polygon shapes.
    If multiple shapes exist, they are merged as foreground.

    Expected minimal structure:
    {
      "imageHeight": H, "imageWidth": W,
      "shapes": [
        {"label": "...", "points": [[x1,y1], [x2,y2], ...], "shape_type": "polygon"},
        ...
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If image size is present, use it; otherwise infer from out_size
    H = data.get("imageHeight", out_size[0])
    W = data.get("imageWidth", out_size[1])

    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if not pts:
            continue
        # Fill polygon as foreground=1
        draw.polygon([tuple(p) for p in pts], outline=1, fill=1)

    # Resize to model size and add channel dimension
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
    images, masks = [], []
    paired = 0
    skipped = 0

    for img_path in pngs:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            skipped += 1
            continue

        # Image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_size[1], img_size[0]))  # (W, H)
        img = np.asarray(img, dtype=np.float32) / 255.0

        # Mask (from JSON)
        mask = labelme_json_to_mask(json_path, out_size=img_size)

        images.append(img)
        masks.append(mask)
        paired += 1

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    print(f"Loaded {paired} image-mask pairs from {data_dir}. Skipped (no JSON): {skipped}")
    print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}], Mask unique: {np.unique(masks)}")
    return images, masks


# =========================
# Metrics & Losses
# =========================
def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def iou_coefficient(y_true, y_pred, smooth: float = 1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def combined_bce_dice(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


# =========================
# FCN-8s (VGG-like backbone)
# =========================
def build_fcn8s(input_shape=(256, 256, 3), num_classes=1):
    """
    FCN-8s: VGG-like encoder, 1x1 score maps, skip connections from pool3 & pool4,
    and transposed-conv upsampling to input resolution.
    Output is sigmoid for binary segmentation.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder (VGG-like)
    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    pool1 = layers.MaxPooling2D(2)(x)  # 1/2

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(pool1)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    pool2 = layers.MaxPooling2D(2)(x)  # 1/4

    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(pool2)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    pool3 = layers.MaxPooling2D(2)(x)  # 1/8

    # Block 4
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(pool3)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    pool4 = layers.MaxPooling2D(2)(x)  # 1/16

    # Block 5
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(pool4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    pool5 = layers.MaxPooling2D(2)(x)  # 1/32

    # 1x1 conv to get score maps
    score_pool3 = layers.Conv2D(1, 1, padding='same')(pool3)   # stride 8
    score_pool4 = layers.Conv2D(1, 1, padding='same')(pool4)   # stride 16
    score_pool5 = layers.Conv2D(1, 1, padding='same')(pool5)   # stride 32

    # Upsample pool5 score by 2 -> stride 16, then add pool4 score
    upscore2 = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False)(score_pool5)
    fuse_pool4 = layers.Add()([upscore2, score_pool4])

    # Upsample fuse by 2 -> stride 8, then add pool3 score
    upscore_pool4 = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False)(fuse_pool4)
    fuse_pool3 = layers.Add()([upscore_pool4, score_pool3])

    # Upsample by 8 -> back to input stride
    upscore8 = layers.Conv2DTranspose(1, kernel_size=16, strides=8, padding='same', use_bias=False)(fuse_pool3)

    outputs = layers.Activation('sigmoid', name="mask")(upscore8)
    model = models.Model(inputs, outputs, name="FCN8s")
    return model


# =========================
# Plot history
# =========================
def plot_training_history(history, out_path="fcn_training_history.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend()

    if 'dice_coefficient' in history.history:
        axes[0, 1].plot(history.history['dice_coefficient'], label='Train Dice')
        axes[0, 1].plot(history.history['val_dice_coefficient'], label='Val Dice')
        axes[0, 1].set_title('Dice'); axes[0, 1].legend()
    else:
        axes[0, 1].axis('off')

    if 'iou_coefficient' in history.history:
        axes[1, 0].plot(history.history['iou_coefficient'], label='Train IoU')
        axes[1, 0].plot(history.history['val_iou_coefficient'], label='Val IoU')
        axes[1, 0].set_title('IoU'); axes[1, 0].legend()
    else:
        axes[1, 0].axis('off')

    if 'accuracy' in history.history:
        axes[1, 1].plot(history.history['accuracy'], label='Train Acc')
        axes[1, 1].plot(history.history['val_accuracy'], label='Val Acc')
        axes[1, 1].set_title('Accuracy'); axes[1, 1].legend()
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()


# =========================
# Main
# =========================
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("Loading dataset...")
    images, masks = load_dataset(DATA_DIR, img_size=IMG_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=VAL_SPLIT, random_state=RANDOM_SEED, shuffle=True
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    print("Building FCN-8s model...")
    model = build_fcn8s(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=combined_bce_dice,
        metrics=['accuracy', dice_coefficient, iou_coefficient]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_fcn8s.keras",
            save_best_only=True,
            monitor="val_dice_coefficient", mode="max", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient", patience=15, mode="max",
            restore_best_weights=True, verbose=1
        )
    ]

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("Saving final model...")
    model.save("final_fcn8s.keras")

    print("Plotting training curves...")
    plot_training_history(history, out_path="fcn_training_history.png")

    print("Evaluating on validation set...")
    val_loss, val_acc, val_dice, val_iou = model.evaluate(X_val, y_val, verbose=0)
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

    print("\nDone. Best model: best_fcn8s.keras | Final: final_fcn8s.keras")

if __name__ == "__main__":
    main()
