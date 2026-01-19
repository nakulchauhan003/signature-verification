#!/usr/bin/env python3
"""
Signature Verification Testing Script
Tests the trained Siamese model for signature verification
"""

import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, '.')

# Import configuration
from utils.config import SIAMESE_MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH

# Import model builder
from models.siamese_model import build_siamese_model


# ============================================
# STEP 1: PREPROCESSING FUNCTION
# ============================================
def preprocess_signature(image_path):
    """
    Preprocess a signature image for model input
    
    Args:
        image_path: Path to the signature image
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to model's expected input size
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values to [0, 1] range
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension (128, 128) -> (128, 128, 1)
    img = np.expand_dims(img, axis=-1)
    
    return img


# ============================================
# STEP 2: LOAD TRAINED MODEL
# ============================================
def load_trained_model():
    """
    Load the trained Siamese model from disk
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(SIAMESE_MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at: {SIAMESE_MODEL_SAVE_PATH}")
    
    print(f"📦 Loading model from: {SIAMESE_MODEL_SAVE_PATH}")
    
    # Rebuild the model architecture
    print("   Building model architecture...")
    model = build_siamese_model()
    
    # Load the trained weights
    print("   Loading trained weights...")
    model.load_weights(SIAMESE_MODEL_SAVE_PATH)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    return model


# ============================================
# STEP 3: COMPUTE SIMILARITY SCORE
# ============================================
def compute_similarity(model, img1_path, img2_path):
    """
    Compare two signatures and compute similarity score
    
    Args:
        model: Trained Siamese model
        img1_path: Path to first signature (reference/genuine)
        img2_path: Path to second signature (test signature)
        
    Returns:
        tuple: (distance, similarity_percentage, is_genuine)
    """
    # Preprocess both images
    img1 = preprocess_signature(img1_path)
    img2 = preprocess_signature(img2_path)
    
    # Add batch dimension: (128, 128, 1) -> (1, 128, 128, 1)
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Get model prediction (Euclidean distance)
    distance = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    
    # Convert distance to similarity percentage
    # Lower distance = Higher similarity
    # Distance typically ranges from 0 (identical) to ~1.5 (very different)
    similarity_percentage = max(0, min(100, (1 - distance) * 100))
    
    # Decision threshold (you can adjust this)
    THRESHOLD = 70.0  # If similarity >= 70%, consider it genuine
    is_genuine = similarity_percentage >= THRESHOLD
    
    return distance, similarity_percentage, is_genuine


# ============================================
# STEP 4: TEST SINGLE PAIR
# ============================================
def test_single_pair(model, img1_path, img2_path, expected_result="unknown"):
    """
    Test a single pair of signatures
    
    Args:
        model: Trained model
        img1_path: Path to reference signature
        img2_path: Path to test signature
        expected_result: "genuine", "forged", or "unknown"
    """
    print(f"\n{'='*60}")
    print(f"🔍 Testing Signature Pair")
    print(f"{'='*60}")
    print(f"Reference: {os.path.basename(img1_path)}")
    print(f"Test:      {os.path.basename(img2_path)}")
    print(f"Expected:  {expected_result.upper()}")
    print(f"{'-'*60}")
    
    # Compute similarity
    distance, similarity, is_genuine = compute_similarity(model, img1_path, img2_path)
    
    # Display results
    print(f"📊 Results:")
    print(f"   Distance:   {distance:.4f}")
    print(f"   Similarity: {similarity:.2f}%")
    print(f"{'-'*60}")
    
    # Verdict
    if is_genuine:
        verdict = "✅ GENUINE"
        confidence = similarity
    else:
        verdict = "❌ FORGED"
        confidence = 100 - similarity
    
    print(f"🎯 Verdict: {verdict} (Confidence: {confidence:.2f}%)")
    
    # Check if prediction matches expectation
    if expected_result != "unknown":
        expected_genuine = (expected_result.lower() == "genuine")
        if is_genuine == expected_genuine:
            print(f"✅ Correct prediction!")
        else:
            print(f"❌ Incorrect prediction (expected {expected_result})")
    
    return distance, similarity, is_genuine


# ============================================
# STEP 5: RUN COMPREHENSIVE TESTS
# ============================================
def run_verification_tests():
    """
    Run comprehensive verification tests on the dataset
    """
    print("\n" + "="*60)
    print("🚀 SIGNATURE VERIFICATION SYSTEM - MODEL TESTING")
    print("="*60)
    
    # Load the trained model
    try:
        model = load_trained_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Setup dataset paths
    dataset_path = "dataset/person1"
    genuine_path = os.path.join(dataset_path, "genuine")
    forged_path = os.path.join(dataset_path, "forged")
    
    # Check if paths exist
    if not os.path.exists(genuine_path):
        print(f"❌ Genuine folder not found: {genuine_path}")
        return
    
    if not os.path.exists(forged_path):
        print(f"⚠️  Forged folder not found: {forged_path}")
        forged_path = None
    
    # Get image files
    genuine_files = sorted([f for f in os.listdir(genuine_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n📁 Dataset Information:")
    print(f"   Genuine signatures: {len(genuine_files)}")
    
    if forged_path:
        forged_files = sorted([f for f in os.listdir(forged_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"   Forged signatures:  {len(forged_files)}")
    else:
        forged_files = []
    
    if len(genuine_files) < 2:
        print("❌ Need at least 2 genuine signatures for testing")
        return
    
    # ========================================
    # TEST 1: Genuine vs Genuine
    # ========================================
    print(f"\n{'#'*60}")
    print("TEST 1: GENUINE vs GENUINE (Should be SIMILAR)")
    print(f"{'#'*60}")
    
    img1_path = os.path.join(genuine_path, genuine_files[0])
    img2_path = os.path.join(genuine_path, genuine_files[1])
    
    test_single_pair(model, img1_path, img2_path, expected_result="genuine")
    
    # ========================================
    # TEST 2: Genuine vs Forged
    # ========================================
    if len(forged_files) > 0:
        print(f"\n{'#'*60}")
        print("TEST 2: GENUINE vs FORGED (Should be DIFFERENT)")
        print(f"{'#'*60}")
        
        img1_path = os.path.join(genuine_path, genuine_files[0])
        img2_path = os.path.join(forged_path, forged_files[0])
        
        test_single_pair(model, img1_path, img2_path, expected_result="forged")
    
    # ========================================
    # TEST 3: Multiple Genuine Comparisons
    # ========================================
    if len(genuine_files) >= 3:
        print(f"\n{'#'*60}")
        print("TEST 3: MULTIPLE GENUINE COMPARISONS")
        print(f"{'#'*60}")
        
        similarities = []
        for i in range(min(3, len(genuine_files) - 1)):
            img1_path = os.path.join(genuine_path, genuine_files[0])
            img2_path = os.path.join(genuine_path, genuine_files[i + 1])
            
            _, similarity, _ = compute_similarity(model, img1_path, img2_path)
            similarities.append(similarity)
            print(f"   {genuine_files[0]} vs {genuine_files[i+1]}: {similarity:.2f}%")
        
        avg_similarity = np.mean(similarities)
        print(f"\n   Average Similarity: {avg_similarity:.2f}%")
    
    # ========================================
    # TEST 4: Multiple Forged Comparisons
    # ========================================
    if len(forged_files) >= 2:
        print(f"\n{'#'*60}")
        print("TEST 4: MULTIPLE FORGED COMPARISONS")
        print(f"{'#'*60}")
        
        similarities = []
        for i in range(min(3, len(forged_files))):
            img1_path = os.path.join(genuine_path, genuine_files[0])
            img2_path = os.path.join(forged_path, forged_files[i])
            
            _, similarity, _ = compute_similarity(model, img1_path, img2_path)
            similarities.append(similarity)
            print(f"   {genuine_files[0]} vs {forged_files[i]}: {similarity:.2f}%")
        
        avg_similarity = np.mean(similarities)
        print(f"\n   Average Similarity: {avg_similarity:.2f}%")
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("📋 TESTING SUMMARY")
    print(f"{'='*60}")
    print("✅ Model is working correctly if:")
    print("   • Genuine vs Genuine: Similarity > 70%")
    print("   • Genuine vs Forged:  Similarity < 50%")
    print(f"\n💡 Threshold: 70% (adjustable in config)")
    print(f"{'='*60}\n")


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    try:
        run_verification_tests()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()