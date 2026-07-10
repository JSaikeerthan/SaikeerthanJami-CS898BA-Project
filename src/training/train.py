"""
train.py

Train the baseline CNN.
"""
import json
from src.models.cnn import build_cnn
from src.training.data_loader import load_datasets
from src.training.trainer import train_model
from pathlib import Path

def main():

    print("=" * 60)
    print("Plant Disease Detection")
    print("Baseline CNN Training")
    print("=" * 60)

    train_ds, val_ds, class_names = load_datasets()

    # Save class names for inference
    metadata_dir = Path("outputs/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "class_names.json", "w") as file:
        json.dump(class_names, file, indent=4)

    print(f"\nClasses: {len(class_names)}")

    model = build_cnn(len(class_names))

    model.summary()

    train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=5,
    )


if __name__ == "__main__":
    main()