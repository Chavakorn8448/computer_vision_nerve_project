import os
import random
import shutil
from pathlib import Path

# --- Config ---
input_folder = Path("nerve_dataset")    # where your .png/.json live
output_train = Path("dataset_train")    # output: training set
output_test  = Path("dataset_test")     # output: test set
train_ratio = 0.8
random_seed = 42                        # set to None for non-reproducible shuffling
REQUIRE_JSON = True                     # if True, only keep PNGs that have a matching .json

# --- Setup ---
output_train.mkdir(parents=True, exist_ok=True)
output_test.mkdir(parents=True, exist_ok=True)

# Gather PNGs
png_files = [p for p in input_folder.iterdir() if p.is_file() and (p.suffix.lower() == ".png" or p.suffix.lower() == ".jpg")]

# Keep only pairs if required
if REQUIRE_JSON:
    paired_pngs = []
    missing_json = 0
    for png in png_files:
        json_path = png.with_suffix(".json")
        if json_path.exists():
            paired_pngs.append(png)
        else:
            missing_json += 1
    png_files = paired_pngs
else:
    missing_json = 0

# Shuffle
if random_seed is not None:
    random.seed(random_seed)
random.shuffle(png_files)

# Split
split_index = int(train_ratio * len(png_files))
train_pngs = png_files[:split_index]
test_pngs = png_files[split_index:]

def copy_pair(png_path: Path, dst_dir: Path):
    """Copy the .png and its paired .json (if exists) to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy PNG
    shutil.copy2(png_path, dst_dir / png_path.name)
    # Copy JSON if present
    json_src = png_path.with_suffix(".json")
    if json_src.exists():
        shutil.copy2(json_src, dst_dir / json_src.name)

# Copy files
for p in train_pngs:
    copy_pair(p, output_train)

for p in test_pngs:
    copy_pair(p, output_test)

print(f"Total PNGs considered: {len(train_pngs) + len(test_pngs)}")
print(f"Training images (PNG): {len(train_pngs)}")
print(f"Testing images  (PNG): {len(test_pngs)}")
if REQUIRE_JSON:
    print(f"Skipped PNGs without matching JSON: {missing_json}")
else:
    print("Note: Some PNGs may not have matching JSON (included).")
