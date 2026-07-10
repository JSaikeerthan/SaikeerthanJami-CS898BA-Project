"""
visualize.py

Generate training accuracy and loss plots.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt


def plot_training_history():
    """
    Generate accuracy and loss plots from history.json.
    """

    history_path = Path("outputs/history/history.json")

    with open(history_path, "r") as file:
        history = json.load(file)

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Training")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png")
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png")
    plt.close()

    print("Training figures saved.")