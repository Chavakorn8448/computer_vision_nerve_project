import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

input_dir = Path("dataset_test")   # folder with .png
output_mask_dir = Path("dataset_test/masks")
output_mask_dir.mkdir(parents=True, exist_ok=True)

for json_file in input_dir.glob("*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            pts = np.array(shape["points"], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)  # fill with 1s

    # Save mask as PNG (0 background, 255 nerve)
    mask_path = output_mask_dir / (json_file.stem + "_mask.png")
    Image.fromarray(mask*255).save(mask_path)

print("Mask generation complete!")
