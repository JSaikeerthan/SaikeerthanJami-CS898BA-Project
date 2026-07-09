"""
Demonstration of image preprocessing steps.
"""

from pathlib import Path

import cv2

from src.preprocessing.resize import resize_image
from src.preprocessing.denoise import gaussian_blur

# Load sample image
dataset = Path("data/raw/PlantVillage/color")

first_class = sorted(
    [folder for folder in dataset.iterdir() if folder.is_dir()]
)[0]

image_path = next(first_class.glob("*"))

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply preprocessing
resized = resize_image(image)
blurred = gaussian_blur(resized)

# Save outputs
output_dir = Path("assets/preprocessing")
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output_dir / "original.png"),
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    str(output_dir / "resized.png"),
    cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    str(output_dir / "gaussian_blur.png"),
    cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
)

print("Preprocessing images saved successfully.")