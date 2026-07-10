"""
evaluate.py

Evaluate the trained CNN model on the validation dataset.
"""

from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from src.training.data_loader import load_datasets

MODEL_PATH = "outputs/models/baseline_cnn.keras"


def main():

    model = tf.keras.models.load_model(MODEL_PATH)

    _, val_ds, class_names = load_datasets(batch_size=16)

    y_true = []
    y_pred = []

    for images, labels in tqdm(val_ds, desc="Evaluating"):

        predictions = model.predict(images, verbose=0)

        predicted = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted)

    print("\nClassification Report")
    print("-" * 60)

    report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    )
    
    print("\nClassification Report")
    print("-" * 60)
    print(report)
    
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "classification_report.txt"
    with open(report_path, "w") as file:
        file.write(report)
        
    print(f"\nClassification report saved to: {report_path.resolve()}")

    cm = confusion_matrix(y_true, y_pred)

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(20, 20))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
        )
    
    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
        )
    
    fig.tight_layout()
    
    figure_path = output_dir / "confusion_matrix.png"
    
    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
        )
    
    plt.close(fig)
    
    print(f"\nConfusion matrix saved to: {figure_path.resolve()}")

if __name__ == "__main__":
    main()