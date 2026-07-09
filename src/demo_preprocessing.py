"""
Demonstration of the image preprocessing pipeline.
"""

from pathlib import Path

import cv2

from src.preprocessing.resize import resize_image
from src.preprocessing.denoise import gaussian_blur
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.hsv import rgb_to_hsv
from src.preprocessing.segmentation import segment_leaf
from src.preprocessing.pipeline import preprocess_image

# ----------------------------------------------------
# Load sample image
# ----------------------------------------------------

dataset = Path("data/raw/PlantVillage/color")

first_class = sorted(
    [folder for folder in dataset.iterdir() if folder.is_dir()]
)[0]

image_path = next(first_class.glob("*"))

print(f"Using image: {image_path.name}")

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Unable to load image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------
# Apply preprocessing pipeline
# ----------------------------------------------------

resized = resize_image(image)

blurred = gaussian_blur(resized)

enhanced = apply_clahe(blurred)

hsv_image = rgb_to_hsv(enhanced)

segmented = segment_leaf(enhanced)

final_image = preprocess_image(image)

# Convert HSV back to RGB ONLY for visualization
hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# ----------------------------------------------------
# Save outputs
# ----------------------------------------------------

output_dir = Path("assets/preprocessing")
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output_dir / "original.png"),
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
)

cv2.imwrite(
    str(output_dir / "resized.png"),
    cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
)

cv2.imwrite(
    str(output_dir / "gaussian_blur.png"),
    cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR),
)

cv2.imwrite(
    str(output_dir / "clahe.png"),
    cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
)

cv2.imwrite(
    str(output_dir / "hsv.png"),
    cv2.cvtColor(hsv_rgb, cv2.COLOR_RGB2BGR),
)

cv2.imwrite(
    str(output_dir / "segmented.png"),
    cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR),
)

final_output = (final_image * 255).astype("uint8")

cv2.imwrite(
    str(output_dir / "final_pipeline.png"),
    cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR),
)

print("\nPreprocessing images saved successfully!")
print(f"Saved images to: {output_dir}")