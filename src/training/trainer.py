"""
trainer.py

Training utilities.
"""

from pathlib import Path
import json

import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    epochs: int = 5,
    model_name: str = "baseline_cnn",
):
    """
    Train a TensorFlow model.

    Args:
        model:
            TensorFlow model.

        train_ds:
            Training dataset.

        val_ds:
            Validation dataset.

        epochs:
            Number of training epochs.

        model_name:
            Name used when saving the trained model and history.

    Returns:
        Training history.
    """

    # Create output directories
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/history").mkdir(parents=True, exist_ok=True)

    # Save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"outputs/models/{model_name}.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # Stop training if validation loss stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            checkpoint,
            early_stop,
        ],
    )

    # Save training history
    history_path = Path("outputs/history") / f"{model_name}_history.json"

    with open(history_path, "w") as file:
        json.dump(history.history, file, indent=4)

    print(f"\nTraining history saved to: {history_path.resolve()}")

    return history