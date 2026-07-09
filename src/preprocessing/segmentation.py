"""
segmentation.py

Leaf segmentation using HSV color thresholding.
"""

import cv2
import numpy as np


def segment_leaf(image: np.ndarray) -> np.ndarray:
    """
    Segment the leaf using HSV thresholding.

    Args:
        image:
            Input RGB image.

    Returns:
        RGB image with background removed.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Green color range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segmented = cv2.bitwise_and(image, image, mask=mask)

    return segmented