"""
Generate training accuracy and loss plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

history_path = Path("outputs/history/training_history.json")

with open(history_path, "r") as file:
    history = json.load(file)

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "accuracy.png", dpi=300)
plt.close()

# Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(history["loss"], marker="o", label="Training Loss")
plt.plot(history["val_loss"], marker="o", label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "loss.png", dpi=300)
plt.close()

print("Training curves generated successfully.")