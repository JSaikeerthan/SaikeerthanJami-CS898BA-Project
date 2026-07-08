"""
dataset.py

Utility functions for loading the PlantVillage dataset.
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import RAW_DATASET_DIR, IMAGE_SIZE


def load_dataset():
    """
    Load all images and labels from the PlantVillage dataset.

    Returns:
        images (numpy.ndarray):
            Array containing all images.

        labels (numpy.ndarray):
            Corresponding class labels.

        class_names (list):
            List of class names.
    """

    images = []
    labels = []

    dataset_path = Path(RAW_DATASET_DIR)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    class_names = sorted(
        [
            folder.name
            for folder in dataset_path.iterdir()
            if folder.is_dir()
        ]
    )

    print("=" * 50)
    print("Loading PlantVillage Dataset")
    print("=" * 50)

    for class_name in class_names:

        class_folder = dataset_path / class_name

        image_files = list(class_folder.glob("*"))

        for image_path in tqdm(
            image_files,
            desc=f"Loading {class_name}"
        ):

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(
                image,
                IMAGE_SIZE
            )

            images.append(image)
            labels.append(class_name)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels)

    print("\nDataset Summary")
    print("-" * 50)
    print(f"Classes      : {len(class_names)}")
    print(f"Images       : {len(images)}")
    print(f"Image Shape  : {images.shape}")
    print("-" * 50)

    return images, labels, class_names