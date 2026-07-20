"""
model_comparison.py

Generate a comparison chart for the models.
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    models = [
        "Baseline CNN",
        "EfficientNetB0",
    ]

    accuracy = [
        90.57,
        97.88,
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(models, accuracy)

    plt.ylim(85, 100)

    plt.ylabel("Validation Accuracy (%)")

    plt.title("Model Performance Comparison")

    for bar, value in zip(bars, accuracy):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.2f}%",
            ha="center",
            fontsize=11,
        )

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_comparison.png",
        dpi=300,
    )

    plt.close()

    print("Model comparison figure saved.")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()