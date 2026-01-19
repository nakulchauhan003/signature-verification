#!/usr/bin/env python3
"""
Comprehensive testing across all persons in dataset
"""

import sys
sys.path.insert(0, '.')

from test_model import load_trained_model, compute_similarity
import os
import numpy as np

def test_all_persons():
    """Test model on all persons in dataset"""
    
    print("="*60)
    print("COMPREHENSIVE SIGNATURE VERIFICATION TEST")
    print("="*60)
    
    model = load_trained_model()
    dataset_path = "dataset"
    
    all_genuine_sims = []
    all_forged_sims = []
    
    persons = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    for person in persons:
        person_path = os.path.join(dataset_path, person)
        genuine_path = os.path.join(person_path, "genuine")
        forged_path = os.path.join(person_path, "forged")
        
        if not os.path.exists(genuine_path):
            continue
        
        genuine_files = [os.path.join(genuine_path, f) 
                        for f in os.listdir(genuine_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        forged_files = []
        if os.path.exists(forged_path):
            forged_files = [os.path.join(forged_path, f)
                           for f in os.listdir(forged_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n{person}:")
        print(f"  Genuine: {len(genuine_files)}, Forged: {len(forged_files)}")
        
        # Test genuine pairs
        if len(genuine_files) >= 2:
            for i in range(min(5, len(genuine_files) - 1)):
                _, sim, _ = compute_similarity(model, genuine_files[0], genuine_files[i+1])
                all_genuine_sims.append(sim)
        
        # Test forged pairs
        if len(forged_files) > 0:
            for i in range(min(5, len(forged_files))):
                _, sim, _ = compute_similarity(model, genuine_files[0], forged_files[i])
                all_forged_sims.append(sim)
    
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Total genuine pairs tested: {len(all_genuine_sims)}")
    print(f"Total forged pairs tested: {len(all_forged_sims)}")
    print(f"\nGenuine similarity: {np.mean(all_genuine_sims):.2f}% (±{np.std(all_genuine_sims):.2f})")
    print(f"Forged similarity: {np.mean(all_forged_sims):.2f}% (±{np.std(all_forged_sims):.2f})")
    print("="*60)

if __name__ == "__main__":
    test_all_persons()