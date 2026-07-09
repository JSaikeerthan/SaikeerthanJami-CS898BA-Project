"""
pipeline.py

Complete image preprocessing pipeline for Plant Disease Detection.
"""

import numpy as np

from src.preprocessing.resize import resize_image
from src.preprocessing.denoise import gaussian_blur
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.segmentation import segment_leaf


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply the complete preprocessing pipeline.

    Pipeline:
        1. Resize
        2. Gaussian Blur
        3. CLAHE
        4. Leaf Segmentation
        5. Normalize

    Args:
        image:
            Input RGB image.

    Returns:
        Preprocessed image normalized to [0,1].
    """

    image = resize_image(image)

    image = gaussian_blur(image)

    image = apply_clahe(image)

    image = segment_leaf(image)

    image = image.astype(np.float32) / 255.0

    return image