# models/resnet152.py

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

def build_resnet152(input_shape=(299, 299, 3), num_classes=3):
    base_model = ResNet152(input_shape=input_shape, include_top=False, weights='imagenet')
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
