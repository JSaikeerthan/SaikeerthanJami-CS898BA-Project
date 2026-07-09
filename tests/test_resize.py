"""
Test the resize preprocessing module.
"""

from pathlib import Path

import cv2

from src.preprocessing.resize import resize_image

# Select the first image from the first class
dataset_root = Path("data/raw/PlantVillage/color")

first_class = sorted(
    [folder for folder in dataset_root.iterdir() if folder.is_dir()]
)[0]

image_path = next(first_class.glob("*"))

print(f"Testing image: {image_path.name}")

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Could not load image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

resized = resize_image(image)

print("Original Shape:", image.shape)
print("Resized Shape :", resized.shape)