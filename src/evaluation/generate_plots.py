"""
generate_plots.py

Generate training accuracy and loss plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HISTORY_PATH = Path("outputs/history/efficientnet_b0_history.json")
FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_history():
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)


def plot_accuracy(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "training_accuracy.png")

    plt.close()


def plot_loss(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "training_loss.png")

    plt.close()


def main():

    history = load_history()

    plot_accuracy(history)

    plot_loss(history)

    print("Training plots saved to:")
    print(FIGURE_DIR.resolve())


if __name__ == "__main__":
    main()