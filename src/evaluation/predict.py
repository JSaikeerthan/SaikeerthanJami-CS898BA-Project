"""
predict.py

Predict the class of a single plant leaf image.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import json

from src.config import IMAGE_SIZE

MODEL_PATH = "outputs/models/efficientnet_b0.keras"
DATASET_DIR = "data/raw/PlantVillage/color"


def load_model():
    """Load the trained CNN model."""
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    """
    Load the class names saved during training.
    """

    with open("outputs/metadata/class_names.json", "r") as file:
        return json.load(file)


def predict_image(image_path: str):

    model = load_model()

    class_names = load_class_names()

    print(f"Image being tested: {image_path}")
    print(f"Ground Truth Folder: {Path(image_path).parent.name}")
    
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE)

# No manual normalization here

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions)
    print(f"Predicted Index: {predicted_index}")
    print(f"Number of Classes: {len(class_names)}")

    confidence = predictions[0][predicted_index]

    print("\nPrediction")
    print("-" * 50)
    print(f"Class      : {class_names[predicted_index]}")
    print(f"Confidence : {confidence:.2%}")
    print("\nFirst 10 classes:")
    for i, name in enumerate(class_names[:10]):
        print(i, name)


if __name__ == "__main__":

    sample_image = next(
        Path(
            "data/raw/PlantVillage/color/Apple___healthy"
        ).glob("*")
    )

    predict_image(str(sample_image))