# data/preprocessing.py

import os
import cv2
import numpy as np
from keras.utils import to_categorical
import random


def verify_image_normalization(image_path: str) -> bool:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img.min() >= 0 and img.max() <= 1


def rotate_image(img, angle):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def add_gaussian_noise(img, mean=0, std=15):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def adjust_brightness_contrast(img, brightness=0.1, contrast=0.1):
    b = random.uniform(-brightness, brightness) * 255
    c = 1.0 + random.uniform(-contrast, contrast)
    return cv2.convertScaleAbs(img, alpha=c, beta=b)


def zoom_image(img, zoom_range=(0.9, 1.1)):
    zoom_factor = random.uniform(*zoom_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(img, (new_w, new_h))

    if zoom_factor < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        padded = cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
        return padded
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h+h, start_w:start_w+w]


def remove_hair_artifacts(img, inpaint=True):
    """Remove hair-like artifacts using morphological black-hat operation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Tuned for long hair lines
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    if inpaint:
        # Inpaint detected hair regions using original image
        inpainted = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        return inpainted
    else:
        return img


def apply_random_augmentation(img):
    aug_img = img.copy()

    if random.random() < 0.3:
        aug_img = cv2.flip(aug_img, 1)
    if random.random() < 0.3:
        aug_img = cv2.flip(aug_img, 0)
    if random.random() < 0.4:
        aug_img = add_gaussian_noise(aug_img)
    if random.random() < 0.4:
        aug_img = adjust_brightness_contrast(aug_img)
    if random.random() < 0.3:
        aug_img = apply_clahe(aug_img)
    if random.random() < 0.3:
        aug_img = zoom_image(aug_img)

    return aug_img


def load_and_augment_images(info_df, image_dir, target_size=(299, 299), angle_step=6, no_angles=360):
    """Load, resize, clean, rotate, and augment mammographic images."""

    X, Y = [], []
    label_map = {'B': 0, 'M': 1, 'N': 2}

    for _, row in info_df.iterrows():
        image_id = row['REFNUM']
        label = row['SEVERITY']
        if label not in label_map:
            continue

        image_path = os.path.join(image_dir, f"{image_id}.pgm")
        img = cv2.imread(image_path)
        if img is None:
            continue

        img = cv2.resize(img, target_size)

        # 💥 Hair removal preprocessing
        img = remove_hair_artifacts(img)

        for angle in range(0, no_angles, angle_step):
            rotated = rotate_image(img, angle)
            X.append(rotated)
            Y.append(label_map[label])

            # Apply N different augmentations per rotated image
            for _ in range(2):
                augmented = apply_random_augmentation(rotated)
                X.append(augmented)
                Y.append(label_map[label])

    X = np.array(X)
    Y = to_categorical(np.array(Y), num_classes=3)
    return X, Y
