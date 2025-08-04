# config.py

import os

class Config:
    # Root directory for dataset and normalized images
    DATA_ROOT = "/project/hussainsyed/adamias"
    IMAGE_DIR = os.path.join(DATA_ROOT, "all-mias-norma")

    # Directory to store trained models
    MODEL_SAVE_PATH = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Training configuration
    IMAGE_SHAPE = (299, 299, 3)
    NUM_CLASSES = 3
    BATCH_SIZE = 256
    EPOCHS = 100

    # For label mapping
    LABELS = {0: 'B', 1: 'M', 2: 'N'}

