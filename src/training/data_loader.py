"""
TensorFlow dataset loader.
"""

import tensorflow as tf

from src.config import IMAGE_SIZE

DATASET_DIR = "data/raw/PlantVillage/color"


def load_datasets(batch_size: int = 16):
    """
    Load training and validation datasets.

    Args:
        batch_size:
            Number of images per batch.

    Returns:
        train_ds:
            Training dataset.

        val_ds:
            Validation dataset.

        class_names:
            List of class names.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    # Do NOT cache the entire dataset in RAM.
    train_ds = (
        train_ds
        .shuffle(1000)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, class_names