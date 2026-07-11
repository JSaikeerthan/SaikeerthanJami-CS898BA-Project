"""
Generate sample predictions from the trained CNN.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.training.data_loader import load_datasets

MODEL_PATH = "outputs/models/baseline_cnn.keras"


def main():

    model = tf.keras.models.load_model(MODEL_PATH)

    _, val_ds, class_names = load_datasets(batch_size=9)

    images, labels = next(iter(val_ds))

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):

        ax.imshow(images[i].numpy().astype("uint8"))

        true_name = class_names[labels[i]].replace("___", " : ").replace("_", " ")
        pred_name = class_names[predicted_labels[i]].replace("___", " : ").replace("_", " ")

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {pred_name}\n"
            f"Conf: {confidences[i]*100:.1f}%",
            fontsize=8,
        )

        ax.axis("off")

    plt.tight_layout()

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "sample_predictions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Sample predictions saved.")


if __name__ == "__main__":
    main()