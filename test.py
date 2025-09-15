from pathlib import Path

# Directories
train_dir = Path("dataset_train/masks")
test_dir = Path("dataset_test/masks")

# Count PNG files
num_train_files = len(list(train_dir.glob("*.png")))
num_test_files = len(list(test_dir.glob("*.png")))

print(f"Total images in dataset_train: {num_train_files}")
print(f"Total images in dataset_test: {num_test_files}")
