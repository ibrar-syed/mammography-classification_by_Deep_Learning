# data/preprocessing.py

import os
import cv2
import numpy as np
from keras.utils import to_categorical

def verify_image_normalization(image_path: str) -> bool:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img.min() >= 0 and img.max() <= 1

def rotate_image(img, angle):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def load_and_augment_images(info_df, image_dir, target_size=(299, 299), angle_step=6, no_angles=360):
    X, Y = [], []
    for i, row in info_df.iterrows():
        image_id = row['REFNUM']
        label = row['SEVERITY']
        label_map = {'B': 0, 'M': 1, 'N': 2}

        if label not in label_map:
            continue

        image_path = os.path.join(image_dir, f"{image_id}.pgm")
        img = cv2.imread(image_path)
        if img is None:
            continue

        img = cv2.resize(img, target_size)

        for angle in range(0, no_angles, angle_step):
            rotated = rotate_image(img, angle)
            X.append(rotated)
            Y.append(label_map[label])

    X = np.array(X)
    Y = to_categorical(np.array(Y), num_classes=3)
    return X, Y
