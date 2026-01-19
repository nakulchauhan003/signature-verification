#!/usr/bin/env python3
"""
Test different threshold values to find optimal setting
"""

import sys
sys.path.insert(0, '.')

from test_model import load_trained_model, compute_similarity
import os

def test_thresholds():
    """Test multiple threshold values"""
    
    # Load model
    print("Loading model...")
    model = load_trained_model()
    
    # Test data paths
    genuine_path = "dataset/person1/genuine"
    forged_path = "dataset/person1/forged"
    
    genuine_files = sorted([f for f in os.listdir(genuine_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    forged_files = sorted([f for f in os.listdir(forged_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Collect genuine pair similarities
    genuine_similarities = []
    for i in range(min(10, len(genuine_files) - 1)):
        img1 = os.path.join(genuine_path, genuine_files[0])
        img2 = os.path.join(genuine_path, genuine_files[i + 1])
        _, sim, _ = compute_similarity(model, img1, img2)
        genuine_similarities.append(sim)
    
    # Collect forged pair similarities
    forged_similarities = []
    for i in range(min(10, len(forged_files))):
        img1 = os.path.join(genuine_path, genuine_files[0])
        img2 = os.path.join(forged_path, forged_files[i])
        _, sim, _ = compute_similarity(model, img1, img2)
        forged_similarities.append(sim)
    
    print("\n" + "="*60)
    print("THRESHOLD TESTING")
    print("="*60)
    
    # Test different thresholds
    thresholds = [50, 55, 60, 65, 70, 75, 80, 85]
    
    print(f"\nGenuine pairs tested: {len(genuine_similarities)}")
    print(f"Forged pairs tested: {len(forged_similarities)}")
    print(f"\nGenuine avg: {sum(genuine_similarities)/len(genuine_similarities):.2f}%")
    print(f"Forged avg: {sum(forged_similarities)/len(forged_similarities):.2f}%")
    
    print("\n" + "-"*60)
    print(f"{'Threshold':<12} {'Genuine OK':<12} {'Forged OK':<12} {'Accuracy':<12}")
    print("-"*60)
    
    for threshold in thresholds:
        genuine_correct = sum(1 for s in genuine_similarities if s >= threshold)
        forged_correct = sum(1 for s in forged_similarities if s < threshold)
        
        total_correct = genuine_correct + forged_correct
        total_tests = len(genuine_similarities) + len(forged_similarities)
        accuracy = (total_correct / total_tests) * 100
        
        print(f"{threshold}%{'':<8} {genuine_correct}/{len(genuine_similarities)}{'':<8} "
              f"{forged_correct}/{len(forged_similarities)}{'':<8} {accuracy:.1f}%")
    
    print("="*60)
    print("\n💡 Recommendation: Choose threshold with highest accuracy")
    print("   while maintaining good genuine detection rate.\n")

if __name__ == "__main__":
    test_thresholds()