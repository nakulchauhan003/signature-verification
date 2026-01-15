import os
import random
import numpy as np
import cv2

from utils.config import DATASET_PATH, IMG_HEIGHT, IMG_WIDTH


def preprocess_image(image_path):
    """
    Load image, resize, normalize, and add channel dimension
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def generate_pairs_all():
    """
    Generate genuine-genuine (label=1) and
    genuine-forged (label=0) pairs for all persons
    """

    X1, X2, Y = [], [], []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        genuine_dir = os.path.join(person_path, "genuine")
        forged_dir = os.path.join(person_path, "forged")

        if not os.path.isdir(genuine_dir):
            continue

        genuine_files = [
            os.path.join(genuine_dir, f)
            for f in os.listdir(genuine_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        forged_files = []
        if os.path.isdir(forged_dir):
            forged_files = [
                os.path.join(forged_dir, f)
                for f in os.listdir(forged_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

        # Need at least 2 genuine signatures
        if len(genuine_files) < 2:
            continue

        # Positive pairs (genuine-genuine)
        for i in range(len(genuine_files) - 1):
            img1 = preprocess_image(genuine_files[i])
            img2 = preprocess_image(genuine_files[i + 1])
            if img1 is not None and img2 is not None:
                X1.append(img1)
                X2.append(img2)
                Y.append(1)

        # Negative pairs (genuine-forged)
        for g in genuine_files:
            if not forged_files:
                continue
            f = random.choice(forged_files)
            img1 = preprocess_image(g)
            img2 = preprocess_image(f)
            if img1 is not None and img2 is not None:
                X1.append(img1)
                X2.append(img2)
                Y.append(0)

    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)

    print("Total pairs generated:", len(Y))
    return X1, X2, Y
