# training/train_model.py
# data/dataset_loader.py
# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint

from data.dataset_loader import load_info_txt
from data.preprocessing import load_and_augment_images
from utils.callbacks import get_callbacks
from utils.visualization import plot_history
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
    print(f"[INFO] Preparing data for model: '{model_name}'")

    df = load_info_txt(Config.DATA_ROOT, include_normal=True)
    X, Y = load_and_augment_images(df, Config.IMAGE_DIR)

    # Flatten labels for stratified split
    y_flat = np.argmax(Y, axis=1)

    # Stratified splitting into train, val, test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_val_idx in sss1.split(X, y_flat):
        x_train, x_temp = X[train_idx], X[test_val_idx]
        y_train, y_temp = Y[train_idx], Y[test_val_idx]
        y_flat_temp = y_flat[test_val_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for val_idx, test_idx in sss2.split(x_temp, y_flat_temp):
        x_val, x_test = x_temp[val_idx], x_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]

    model_builder = MODEL_REGISTRY.get(model_name)
    if not model_builder:
        raise ValueError(f"[ERROR] Model '{model_name}' not found in registry.")

    model = model_builder(input_shape=Config.IMAGE_SHAPE, num_classes=Config.NUM_CLASSES)

    # Create model-specific output dir
    model_output_dir = os.path.join(Config.MODEL_SAVE_PATH, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_output_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    print(f"[INFO] Starting training for {Config.EPOCHS} epochs...")
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks() + [checkpoint],
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] Training completed in {elapsed:.2f} seconds.")

    # Save final model
    final_model_path = os.path.join(model_output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"[INFO] Final model saved at: {final_model_path}")

    # Save training logs
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_output_dir, "history.csv"), index=False)

    # Evaluate on test set
    print(f"[INFO] Evaluating on test data...")
    y_pred = model.predict(x_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_test, axis=1)

    print(classification_report(y_true_class, y_pred_class, target_names=Config.LABELS.values()))
    print(f"Test Accuracy: {accuracy_score(y_true_class, y_pred_class):.4f}")

    # Optional: plot training history
    try:
        plot_history(history)
    except Exception as e:
        print("[WARN] Could not plot training history:", e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., mobilenetv3, xception)")
    args = parser.parse_args()

    run_training(args.model.lower())
