import os
import random
import shutil

# Paths
input_folder = "dataset"        # folder where all your images are
output_train = "dataset_train"  # folder to save training images
output_test = "dataset_test"    # folder to save testing images

# Create output folders if they don’t exist
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# List all files
files = os.listdir(input_folder)
random.shuffle(files)  # shuffle for randomness

# Split index (80% train, 20% test)
split_index = int(0.8 * len(files))

train_files = files[:split_index]
test_files = files[split_index:]

# Copy train files
for f in train_files:
    shutil.copy(os.path.join(input_folder, f), os.path.join(output_train, f))

# Copy test files
for f in test_files:
    shutil.copy(os.path.join(input_folder, f), os.path.join(output_test, f))

print(f"Training images: {len(train_files)}")
print(f"Testing images: {len(test_files)}")
