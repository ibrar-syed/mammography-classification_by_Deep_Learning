# models/mobilenetv3.py

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
