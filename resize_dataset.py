"""
Resize all images in Dataset/ to 256x256 and save as JPEG.
Run this once before training to standardize sizes and reduce storage.

Usage: uv run python resize_dataset.py
"""

import os
from PIL import Image

SIZE = 256
DATASET_DIR = "./Dataset"
QUALITY = 90

count = 0
for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            print(f"Skipping {fpath}")
            continue

        img = img.resize((SIZE, SIZE), Image.LANCZOS)

        # Save as .jpg, remove old file if different extension
        new_name = os.path.splitext(fname)[0] + ".jpg"
        new_path = os.path.join(class_dir, new_name)
        img.save(new_path, "JPEG", quality=QUALITY)

        if fpath != new_path:
            os.remove(fpath)

        count += 1

print(f"Resized {count} images to {SIZE}x{SIZE}")
