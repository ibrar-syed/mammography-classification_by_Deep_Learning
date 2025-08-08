# models/mobilenetv3.py
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
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

def build_mobilenetv3(input_shape=(299, 299, 3), num_classes=3) -> Sequential:
    base_model = MobileNetV3Large(input_shape=input_shape, include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        BatchNormalization(),
        Dense(1024, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dense(512, kernel_initializer='he_uniform'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
