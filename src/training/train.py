"""
train.py

Train the Plant Disease Detection model.
"""

import json
from pathlib import Path

from src.models.efficientnet import build_efficientnet
from src.training.data_loader import load_datasets
from src.training.trainer import train_model


def main():

    print("=" * 60)
    print("Plant Disease Detection")
    print("EfficientNetB0 Transfer Learning")
    print("=" * 60)

    # Load datasets
    train_ds, val_ds, class_names = load_datasets()

    # Save class names for inference
    metadata_dir = Path("outputs/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_dir / "class_names.json", "w") as file:
        json.dump(class_names, file, indent=4)

    print(f"\nClasses: {len(class_names)}")

    # Build model
    model = build_efficientnet(len(class_names))

    # Print model architecture
    model.summary()

    # Train model
    train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=5,
        model_name="efficientnet_b0_finetuned",
    )


if __name__ == "__main__":
    main()