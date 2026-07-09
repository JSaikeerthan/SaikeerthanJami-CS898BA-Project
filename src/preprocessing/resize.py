"""
resize.py

Utility functions for resizing images.
"""

import cv2
import numpy as np

from src.config import IMAGE_SIZE


def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Resize an image to the configured project size.

    Args:
        image:
            Input image as a NumPy array.

    Returns:
        Resized image.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return resized