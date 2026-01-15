import cv2
import numpy as np
from PIL import Image
import os


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess a signature image for the Siamese network
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (width, height)
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Apply noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Enhance contrast using histogram equalization
    image = cv2.equalizeHist(image)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    
    # Add channel dimension (for grayscale)
    image = np.expand_dims(image, axis=-1)
    
    return image


def normalize_signature(image_array):
    """
    Normalize a signature image array
    
    Args:
        image_array (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Normalized image array
    """
    # Ensure values are in the range [0, 1]
    normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    return normalized


def remove_background(image_path, threshold=50):
    """
    Remove background from signature image
    
    Args:
        image_path (str): Path to the image file
        threshold (int): Threshold value for binarization
    
    Returns:
        numpy.ndarray: Image with background removed
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply threshold to create binary image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for the signature
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, contours, 255)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, mask)
    
    return result


def clean_signature(image_path):
    """
    Complete cleaning pipeline for signature images
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        numpy.ndarray: Cleaned image array
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Normalize the image
    image = normalize_signature(image)
    
    return image


if __name__ == "__main__":
    # Test the preprocessing functions
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        processed = preprocess_image(test_image)
        print(f"Image processed successfully. Shape: {processed.shape}")
    else:
        print("Usage: python preprocess.py <image_path>")