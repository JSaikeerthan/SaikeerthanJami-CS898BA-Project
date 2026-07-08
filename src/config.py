from pathlib import Path

# src/
SRC_DIR = Path(__file__).resolve().parent

# Project root
PROJECT_ROOT = SRC_DIR.parent

# Dataset Paths
RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "color"
PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

# Image settings
IMAGE_SIZE = (224, 224)

# Random seed
RANDOM_SEED = 42