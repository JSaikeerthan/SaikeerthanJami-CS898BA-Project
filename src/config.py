from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed"

# Image settings
IMAGE_SIZE = (224, 224)

# Random seed
RANDOM_SEED = 42