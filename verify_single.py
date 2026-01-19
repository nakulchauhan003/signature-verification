#!/usr/bin/env python3
"""
Simple script to verify a single signature pair
Usage: python verify_single.py <reference_signature> <test_signature>
"""

import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, '.')

from utils.config import SIAMESE_MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH
from models.siamese_model import build_siamese_model


def preprocess_signature(image_path):
    """Preprocess a signature image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def verify_signature(reference_path, test_path, threshold=70.0):
    """
    Verify if a test signature matches the reference signature
    
    Args:
        reference_path: Path to reference (genuine) signature
        test_path: Path to test signature
        threshold: Similarity threshold (default: 70%)
        
    Returns:
        dict with verification results
    """
    # Load model
    print("Loading model...")
    model = build_siamese_model()
    model.load_weights(SIAMESE_MODEL_SAVE_PATH)
    print("✅ Model loaded\n")
    
    # Preprocess images
    img1 = preprocess_signature(reference_path)
    img2 = preprocess_signature(test_path)
    
    # Add batch dimension
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Get prediction
    distance = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    similarity = max(0, min(100, (1 - distance) * 100))
    is_genuine = similarity >= threshold
    
    # Return results
    return {
        'distance': distance,
        'similarity': similarity,
        'is_genuine': is_genuine,
        'threshold': threshold,
        'confidence': similarity if is_genuine else (100 - similarity)
    }


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python verify_single.py <reference_signature> <test_signature>")
        print("\nExample:")
        print("  python verify_single.py dataset/person1/genuine/1.jpg dataset/person1/genuine/2.jpg")
        sys.exit(1)
    
    reference_path = sys.argv[1]
    test_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(reference_path):
        print(f"❌ Reference file not found: {reference_path}")
        sys.exit(1)
    
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        sys.exit(1)
    
    print("="*60)
    print("🔍 SIGNATURE VERIFICATION")
    print("="*60)
    print(f"Reference: {reference_path}")
    print(f"Test:      {test_path}")
    print("="*60 + "\n")
    
    # Verify
    result = verify_signature(reference_path, test_path)
    
    # Display results
    print("📊 RESULTS:")
    print(f"   Distance:   {result['distance']:.4f}")
    print(f"   Similarity: {result['similarity']:.2f}%")
    print(f"   Threshold:  {result['threshold']:.2f}%")
    print("-"*60)
    
    if result['is_genuine']:
        print(f"✅ VERDICT: GENUINE")
        print(f"   Confidence: {result['confidence']:.2f}%")
    else:
        print(f"❌ VERDICT: FORGED")
        print(f"   Confidence: {result['confidence']:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()
