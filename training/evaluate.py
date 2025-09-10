# training/evaluate.py,...
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
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tensorflow.keras.models import load_model

from data.dataset_loader import load_info_txt
from data.preprocessing import load_and_augment_images
from config import Config

def evaluate_model(model_path: str):
    df = load_info_txt(Config.DATA_ROOT, include_normal=True)
    _, _, X_test, Y_test = np.split(load_and_augment_images(df, Config.IMAGE_DIR), [0, 0, int(0.8 * len(df))])
    y_true = np.argmax(Y_test, axis=1)

    model = load_model(model_path)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Evaluation Results:")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='weighted'), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='weighted'), 4))
    print("F1 Score :", round(f1_score(y_true, y_pred, average='weighted'), 4))
    print("Kappa    :", round(cohen_kappa_score(y_true, y_pred), 4))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['B', 'M', 'N']))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.h5)")
    args = parser.parse_args()

    evaluate_model(args.model_path)
