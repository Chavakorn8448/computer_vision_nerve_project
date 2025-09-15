from pathlib import Path
from PIL import Image

# --- Config ---
dataset_dir = Path("nerve_dataset")  # Your original dataset folder

# --- Convert JPG to PNG ---
for img_path in dataset_dir.glob("*.jpg"):
    png_path = img_path.with_suffix(".png")  # new PNG path
    # Open JPG and save as PNG
    with Image.open(img_path) as img:
        img.save(png_path)
    # Remove original JPG
    img_path.unlink()

print("Conversion complete! All images are now PNG.")
