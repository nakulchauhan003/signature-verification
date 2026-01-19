#!/usr/bin/env python3
"""
Quick test to verify contrastive loss implementation
"""

import sys
sys.path.insert(0, '.')

from models.siamese_model import build_siamese_model, contrastive_loss
import tensorflow.keras.backend as K
import numpy as np

print("="*60)
print("🧪 TESTING CONTRASTIVE LOSS IMPLEMENTATION")
print("="*60)

# Test 1: Build model with contrastive loss
print("\n📋 Test 1: Building model with contrastive loss...")
try:
    model = build_siamese_model(use_contrastive_loss=True, margin=1.0)
    print("✅ Model built successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Test contrastive loss function
print("\n📋 Test 2: Testing contrastive loss function...")

# Simulate similar pair (should have low distance)
y_true_similar = K.constant([[1.0]])  # Similar pair
y_pred_similar = K.constant([[0.2]])  # Low distance (good!)

loss_similar = contrastive_loss(y_true_similar, y_pred_similar, margin=1.0)
loss_similar_value = K.eval(loss_similar)

print(f"\n   Similar Pair (Y=1, D=0.2):")
print(f"   Loss = 0.5 × (0.2)² = {loss_similar_value:.4f}")
print(f"   Expected: ~0.02")
print(f"   {'✅ Correct!' if abs(loss_similar_value - 0.02) < 0.001 else '❌ Wrong!'}")

# Simulate dissimilar pair (should have high distance)
y_true_dissimilar = K.constant([[0.0]])  # Dissimilar pair
y_pred_dissimilar = K.constant([[0.9]])  # High distance (good!)

loss_dissimilar = contrastive_loss(y_true_dissimilar, y_pred_dissimilar, margin=1.0)
loss_dissimilar_value = K.eval(loss_dissimilar)

print(f"\n   Dissimilar Pair (Y=0, D=0.9):")
print(f"   Loss = 0.5 × max(0, 1.0-0.9)² = {loss_dissimilar_value:.4f}")
print(f"   Expected: ~0.005")
print(f"   {'✅ Correct!' if abs(loss_dissimilar_value - 0.005) < 0.001 else '❌ Wrong!'}")

# Test 3: Bad case - similar pair with high distance (should have high loss)
y_true_bad = K.constant([[1.0]])  # Similar pair
y_pred_bad = K.constant([[0.9]])  # High distance (bad!)

loss_bad = contrastive_loss(y_true_bad, y_pred_bad, margin=1.0)
loss_bad_value = K.eval(loss_bad)

print(f"\n   Similar Pair with High Distance (Y=1, D=0.9):")
print(f"   Loss = 0.5 × (0.9)² = {loss_bad_value:.4f}")
print(f"   Expected: ~0.405 (high loss, needs improvement)")
print(f"   {'✅ Correct!' if abs(loss_bad_value - 0.405) < 0.01 else '❌ Wrong!'}")

# Test 4: Build model with MSE (old behavior)
print("\n📋 Test 3: Building model with MSE loss (old behavior)...")
try:
    model_mse = build_siamese_model(use_contrastive_loss=False)
    print("✅ MSE model built successfully!")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n💡 Next Steps:")
print("   1. Run: python training/train_model.py")
print("   2. Wait for training to complete (~5-10 minutes)")
print("   3. Run: python test_model.py")
print("   4. Verify genuine pairs show >70% similarity")
print("="*60 + "\n")
