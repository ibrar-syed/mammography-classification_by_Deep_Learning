# training/train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split

from data.dataset_loader import load_info_txt
from data.preprocessing import load_and_augment_images
from utils.callbacks import get_callbacks
from config import Config

# Dynamic model loading
from models.mobilenetv3 import build_mobilenetv3
from models.nasnetmobile import build_nasnetmobile
from models.resnetrs import build_resnetrs
from models.xception import build_xception
from models.resnet152 import build_resnet152
from models.densenet201 import build_densenet201

MODEL_REGISTRY = {
    "mobilenetv3": build_mobilenetv3,
    "nasnetmobile": build_nasnetmobile,
    "resnetrs": build_resnetrs,
    "xception": build_xception,
    "resnet152": build_resnet152,
    "densenet201": build_densenet201,
}


def run_training(model_name: str):
    print(f"Loading info for training '{model_name}' model...")

    df = load_info_txt(Config.DATA_ROOT, include_normal=True)
    X, Y = load_and_augment_images(df, Config.IMAGE_DIR)

    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)

    model_builder = MODEL_REGISTRY.get(model_name)
    if not model_builder:
        raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY")

    model = model_builder(input_shape=Config.IMAGE_SHAPE, num_classes=Config.NUM_CLASSES)

    print("Starting training...")
    history = model.fit(
        x_train,
        y_train,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(),
        batch_size=Config.BATCH_SIZE
    )

    model.save(os.path.join(Config.MODEL_SAVE_PATH, f"{model_name}.h5"))
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., mobilenetv3, xception)")
    args = parser.parse_args()

    run_training(args.model)
