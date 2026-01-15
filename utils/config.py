"""
Configuration file for Signature Verification System
This version is fully compatible with the current project codebase.
"""

import os

# ===============================
# Project Paths
# ===============================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
PREPROCESSING_PATH = os.path.join(PROJECT_ROOT, "preprocessing")
TRAINING_PATH = os.path.join(PROJECT_ROOT, "training")
VERIFICATION_PATH = os.path.join(PROJECT_ROOT, "verification")
APP_PATH = os.path.join(PROJECT_ROOT, "app")
UTILS_PATH = os.path.join(PROJECT_ROOT, "utils")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# Ensure important directories exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# ===============================
# Image / Model Configuration
# ===============================

# Target image size (used everywhere)
IMAGE_TARGET_SIZE = (128, 128)   # (height, width)
IMG_HEIGHT = IMAGE_TARGET_SIZE[0]
IMG_WIDTH = IMAGE_TARGET_SIZE[1]
IMG_CHANNELS = 1

# Model input shape
MODEL_INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Model save paths
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, "trained_model.h5")
SIAMESE_MODEL_SAVE_PATH = os.path.join(MODELS_PATH, "siamese_trained.h5")

# ===============================
# Training Configuration
# ===============================

TRAINING_EPOCHS = 5          # keep small for now
BATCH_SIZE = 8               # safe for CPU
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# ===============================
# Verification Configuration
# ===============================

VERIFICATION_THRESHOLD = 0.5
FORGERY_DETECTION_ENABLED = True
DRIFT_DETECTION_ENABLED = True

# ===============================
# Preprocessing Configuration
# ===============================

IMAGE_GRAYSCALE = True
NORMALIZATION_FACTOR = 255.0

# ===============================
# Dataset Configuration
# ===============================

DATASET_SPLIT_RATIO = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15
}

# ===============================
# Performance Configuration
# ===============================

NUM_WORKERS = 2
USE_GPU = False  # keep False unless GPU is confirmed

# ===============================
# Logging Configuration
# ===============================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(RESULTS_PATH, "signature_verification.log")

# ===============================
# Security / Upload Configuration
# ===============================

MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

