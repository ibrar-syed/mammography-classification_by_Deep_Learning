### config.py,......

# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#####
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# utils/metrics.py

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

