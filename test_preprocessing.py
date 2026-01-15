# test_preprocessing.py
import cv2
import numpy as np
from preprocessing.preprocess import preprocess_image
import os

def test_preprocessing():
    # Test with actual signature files from your dataset
    genuine_path = "dataset/person1/Original signatures/1.jpg"
    forged_path = "dataset/person1/Forged signatures/1.jpg"
    
    print("🔍 TESTING PREPROCESSING...")
    print("=" * 50)
    
    # Test genuine signature
    print(f"\n📄 Testing genuine signature: {genuine_path}")
    if os.path.exists(genuine_path):
        processed_genuine = preprocess_image(genuine_path)
        print(f"✅ SUCCESS! Shape: {processed_genuine.shape}")
        print(f"📊 Value range: {processed_genuine.min():.3f} to {processed_genuine.max():.3f}")
        
        # Save processed image to see result
        display_img = (processed_genuine.squeeze() * 255).astype(np.uint8)
        cv2.imwrite("processed_genuine_test.png", display_img)
        print("💾 Saved as: processed_genuine_test.png")
    else:
        print("❌ ERROR: Genuine signature file not found!")
    
    # Test forged signature
    print(f"\n📄 Testing forged signature: {forged_path}")
    if os.path.exists(forged_path):
        processed_forged = preprocess_image(forged_path)
        print(f"✅ SUCCESS! Shape: {processed_forged.shape}")
        print(f"📊 Value range: {processed_forged.min():.3f} to {processed_forged.max():.3f}")
        
        # Save processed image to see result
        display_img = (processed_forged.squeeze() * 255).astype(np.uint8)
        cv2.imwrite("processed_forged_test.png", display_img)
        print("💾 Saved as: processed_forged_test.png")
    else:
        print("❌ ERROR: Forged signature file not found!")
    
    print("\n" + "=" * 50)
    print("🎯 WHAT TO CHECK IN THE SAVED IMAGES:")
    print("• White/clear signature strokes")
    print("• Dark/black background") 
    print("• Smooth, clean lines")
    print("• Good contrast between signature and background")
    print("=" * 50)

if __name__ == "__main__":
    test_preprocessing()