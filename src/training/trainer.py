"""
trainer.py

Training utilities for the baseline CNN.
"""

from pathlib import Path
import json
import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    epochs: int = 5,
):
    """
    Train the CNN model.

    Args:
        model:
            CNN model.

        train_ds:
            Training dataset.

        val_ds:
            Validation dataset.

        epochs:
            Number of training epochs.

    Returns:
        Training history.
    """

    # Ensure output directories exist
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="outputs/models/baseline_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            checkpoint,
            early_stop,
        ],
    )
    
    # Create history directory
    history_dir = Path("outputs/history")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history
    with open(history_dir / "history.json", "w") as file:
        json.dump(history.history, file, indent=4)

    return history