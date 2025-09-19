from CNN_model import build_unet, dice_loss, dice_coefficient
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

def load_data(image_dir, mask_dir, img_size=(256, 256)):
    """Load and preprocess images and masks consistently"""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # Get all image files
    image_paths = sorted([p for p in image_dir.glob("*.png") if not p.name.endswith("_mask.png")])
    
    # Find corresponding mask files
    mask_paths = []
    valid_image_paths = []
    
    for img_path in image_paths:
        mask_path = mask_dir / (img_path.stem + "_mask.png")
        if mask_path.exists():
            mask_paths.append(mask_path)
            valid_image_paths.append(img_path)
    
    print(f"Found {len(valid_image_paths)} image-mask pairs")
    
    images, masks = [], []
    for img_path, mask_path in zip(valid_image_paths, mask_paths):
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img.resize(img_size)) / 255.0
        
        # Load and resize mask
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask = np.array(mask.resize(img_size)) / 255.0  # Normalize to 0-1
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        
        # Ensure binary mask
        mask = (mask > 0.5).astype(np.float32)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Dice coefficient
    axes[0, 1].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[0, 1].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    
    # Accuracy
    axes[1, 0].plot(history.history['accuracy'], label='Train Acc')
    axes[1, 0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1, 0].set_title('Model Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    
    # Learning rate (if using scheduler)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Load data
print("Loading training data...")
images, masks = load_data("dataset_train", "dataset_train/masks")

# Check data shapes
print(f"Images shape: {images.shape}")
print(f"Masks shape: {masks.shape}")
print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
print(f"Mask unique values: {np.unique(masks)}")

# Split data
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42, stratify=None
)

print(f"\nData split:")
print(f"Training images: {train_imgs.shape[0]}")
print(f"Validation images: {val_imgs.shape[0]}")

# Build and compile model
print("\nBuilding model...")
model = build_unet(input_shape=(256, 256, 3), num_classes=1, dropout_rate=0.1)

# Use a combination of dice loss and binary crossentropy
def combined_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=['accuracy', dice_coefficient]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_nerve_unet_model.keras", 
        save_best_only=True, 
        monitor='val_dice_coefficient',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    # Early stopping based on accuracy to prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Stop if val_accuracy does not improve for 5 epochs
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
]

# Train model
print("\nStarting training...")
history = model.fit(
    train_imgs, train_masks,
    validation_data=(val_imgs, val_masks),
    batch_size=4,  # Increased batch size
    epochs=50,     # Increased epochs
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("Model_save_test.keras")
print("Final model saved.")
# Plot training history
plot_training_history(history)

# Final evaluation
print(f"\nFinal evaluation on original validation set (no augmentation):")
val_loss, val_acc, val_dice = model.evaluate(val_imgs, val_masks, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Dice Score: {val_dice:.4f}")

print("\nTraining completed!")
print("Best model saved as: best_nerve_unet_model.keras")
print("Final model saved as: final_nerve_unet_model.keras")
print(f"\nDataset summary:")
print(f"- Original training images: {len(train_imgs)//3}")  # Divide by 3 due to augmentation
print(f"- Augmented training images: {len(train_imgs)}")
print(f"- Validation images: {len(val_imgs)}")
print(f"- Total original dataset: ~{len(images)}")