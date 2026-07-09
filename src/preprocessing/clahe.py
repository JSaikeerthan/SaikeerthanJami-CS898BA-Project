"""
clahe.py

Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).
"""

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE to an RGB image.

    Args:
        image:
            Input RGB image.

        clip_limit:
            Contrast limiting threshold.

        tile_grid_size:
            Size of the grid for histogram equalization.

    Returns:
        Enhanced RGB image.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_channel = clahe.apply(l_channel)

    enhanced = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2RGB
    )

    return enhanced