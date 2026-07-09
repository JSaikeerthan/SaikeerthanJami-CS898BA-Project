"""
denoise.py

Noise reduction using Gaussian Blur.
"""

import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian Blur to reduce image noise.

    Args:
        image:
            Input RGB image.

        kernel_size:
            Size of Gaussian kernel.

        sigma:
            Gaussian sigma value.
            If 0, OpenCV calculates it automatically.

    Returns:
        Blurred image.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    return cv2.GaussianBlur(
        image,
        kernel_size,
        sigma
    )