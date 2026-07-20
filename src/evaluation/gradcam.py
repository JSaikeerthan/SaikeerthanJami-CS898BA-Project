"""
gradcam.py

Generate Grad-CAM visualizations for EfficientNetB0.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import IMAGE_SIZE

MODEL_PATH = "outputs/models/efficientnet_b0.keras"
CLASS_NAMES_PATH = "outputs/metadata/class_names.json"

GRADCAM_DIR = Path("outputs/gradcam")
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load the trained EfficientNet model."""
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    """Load class names."""
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


def load_image(image_path: str):
    """
    Load an image for prediction.
    """

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE)

    image_array = np.expand_dims(image, axis=0)

    return image, image_array


def predict(model, image_array, class_names):
    """
    Predict the class of an image.
    """

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index]

    predicted_class = class_names[predicted_index]

    print("=" * 60)
    print("Prediction")
    print("=" * 60)

    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence:.2%}")

    return predicted_index

def make_gradcam_heatmap(model, image_array, last_conv_layer_name="top_conv"):
    """
    Generate Grad-CAM heatmap.
    """

    # Get EfficientNet backbone
    base_model = model.get_layer("efficientnetb0")

    # Last convolutional layer
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Model from EfficientNet input to last conv layer
    last_conv_model = tf.keras.Model(
        base_model.input,
        last_conv_layer.output,
    )

    # Classifier after top_conv
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    passed_last_conv = False

    for layer in base_model.layers:

        if layer.name == last_conv_layer_name:
            passed_last_conv = True
            continue

        if passed_last_conv:
            x = layer(x)

    # Remaining layers after EfficientNet
    for layer in model.layers[2:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    # Compute gradients
    with tf.GradientTape() as tape:

        conv_outputs = last_conv_model(image_array)

        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)

        predicted_class = tf.argmax(predictions[0])

        class_channel = predictions[:, predicted_class]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.math.reduce_max(heatmap) + 1e-10

    return heatmap.numpy()

def save_gradcam(image, heatmap):
    """
    Save original image, heatmap, and Grad-CAM overlay.
    """

    # Resize heatmap to original image size
    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0]),
    )

    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply color map
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB,
    )

    # Overlay
    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap_color,
        0.4,
        0,
    )

    # Save images
    plt.imsave(
        GRADCAM_DIR / "original.png",
        image,
    )

    plt.imsave(
        GRADCAM_DIR / "heatmap.png",
        heatmap_color,
    )

    plt.imsave(
        GRADCAM_DIR / "overlay.png",
        overlay,
    )

    print("\nGrad-CAM images saved to:")
    print(GRADCAM_DIR.resolve())


def main():

    model = load_model()

    class_names = load_class_names()

    # Change this folder to test another disease
    sample_image = next(
        Path(
            "data/raw/PlantVillage/color/Apple___Black_rot"
        ).glob("*")
    )

    print(f"Testing image: {sample_image.name}")

    image, image_array = load_image(str(sample_image))

    predict(
        model,
        image_array,
        class_names,
    )

    heatmap = make_gradcam_heatmap(
        model,
        image_array,
        last_conv_layer_name="top_conv",
    )

    save_gradcam(
        image,
        heatmap,
    )


if __name__ == "__main__":
    main()