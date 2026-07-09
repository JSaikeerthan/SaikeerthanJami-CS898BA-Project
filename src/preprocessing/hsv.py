"""
hsv.py

Convert RGB images to the HSV color space.
"""

import cv2
import numpy as np


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to HSV.

    Args:
        image:
            Input RGB image.

    Returns:
        HSV image.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)