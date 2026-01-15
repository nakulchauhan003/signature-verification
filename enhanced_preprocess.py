# enhanced_preprocess.py
import cv2
import numpy as np
from PIL import Image
import os

def enhanced_signature_preprocess(image_path, target_size=(128, 128)):
    """
    Enhanced preprocessing specifically designed for signature images
    Handles noise, poor lighting, and background removal
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"📥 Loaded image: {image.shape}")
    
    # Step 1: Initial noise reduction
    # Apply bilateral filter to preserve edges while removing noise
    image = cv2.bilateralFilter(image, 9, 75, 75)
    print("✨ Applied bilateral filtering for noise reduction")
    
    # Step 2: Background estimation and removal
    # Estimate background using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    background = cv2.medianBlur(background, 21)
    
    # Subtract background
    diff = cv2.subtract(background, image)
    print("🧹 Removed background texture")
    
    # Step 3: Adaptive thresholding (handles uneven lighting)
    # This is the key improvement for your lighting issues
    binary = cv2.adaptiveThreshold(
        diff,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,  # Block size - smaller = better for fine details
        5    # Constant subtracted from mean
    )
    print("🎯 Applied adaptive thresholding")
    
    # Step 4: Morphological cleanup
    # Remove small noise and connect broken strokes
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Remove small noise dots
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    # Connect nearby strokes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
    print("🔧 Applied morphological cleanup")
    
    # Step 5: Find and extract signature region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely the signature)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask for signature only
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # Apply mask to get clean signature
        cleaned = cv2.bitwise_and(diff, mask)
        print("✂️ Extracted signature region")
    else:
        # Fallback if no contours found
        cleaned = diff
        print("⚠️ No contours found, using diff image")
    
    # Step 6: Final enhancement
    # Enhance contrast of the signature
    cleaned = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(cleaned.astype(np.uint8))
    print("🌈 Enhanced signature contrast")
    
    # Step 7: Resize to target size
    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    
    # Step 8: Convert to model format (0-1 range, add channel dimension)
    final = resized.astype(np.float32) / 255.0
    final = np.expand_dims(final, axis=-1)
    
    print(f"✅ Final processed shape: {final.shape}")
    return final

def visualize_preprocessing_steps(image_path):
    """
    Show all preprocessing steps for debugging
    """
    # Load original
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply each step and save intermediate results
    steps = [
        ("1_original", original),
        ("2_bilateral_filtered", cv2.bilateralFilter(original, 9, 75, 75)),
    ]
    
    # Background removal step
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    background = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
    background = cv2.medianBlur(background, 21)
    diff = cv2.subtract(background, original)
    steps.append(("3_background_removed", diff))
    
    # Thresholding
    binary = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 15, 5)
    steps.append(("4_adaptive_threshold", binary))
    
    # Morphological operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
    steps.append(("5_morphology_cleaned", cleaned))
    
    # Final enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    steps.append(("6_final_enhanced", enhanced))
    
    # Save all steps
    for name, img in steps:
        if len(img.shape) == 2:  # Grayscale
            cv2.imwrite(f"{name}.png", img)
        else:  # Already colored
            cv2.imwrite(f"{name}.png", img)
    
    print("📸 Saved all preprocessing steps as separate images")

# Test function
def test_enhanced_preprocessing():
    """Test the enhanced preprocessing"""
    genuine_path = "dataset/person1/Original signatures/1.jpg"
    forged_path = "dataset/person1/Forged signatures/1.jpg"
    
    print("🔬 TESTING ENHANCED PREPROCESSING")
    print("=" * 50)
    
    if os.path.exists(genuine_path):
        print(f"\n📄 Processing genuine signature: {genuine_path}")
        processed = enhanced_signature_preprocess(genuine_path)
        display_img = (processed.squeeze() * 255).astype(np.uint8)
        cv2.imwrite("enhanced_genuine_test.png", display_img)
        print("✅ Enhanced genuine signature saved!")
        
        # Also save step-by-step visualization
        visualize_preprocessing_steps(genuine_path)
        print("📊 Preprocessing steps saved as separate images")
    
    if os.path.exists(forged_path):
        print(f"\n📄 Processing forged signature: {forged_path}")
        processed = enhanced_signature_preprocess(forged_path)
        display_img = (processed.squeeze() * 255).astype(np.uint8)
        cv2.imwrite("enhanced_forged_test.png", display_img)
        print("✅ Enhanced forged signature saved!")

if __name__ == "__main__":
    test_enhanced_preprocessing()