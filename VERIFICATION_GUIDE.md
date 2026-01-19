# Signature Verification - Testing & Verification Guide

## 📋 Overview

This guide explains how to test and use the trained Siamese neural network model for signature verification.

## 🔧 Model Architecture

The system uses a **Siamese Neural Network** that:
1. Takes two signature images as input
2. Processes each through a shared CNN (Convolutional Neural Network)
3. Computes the Euclidean distance between the feature vectors
4. Outputs a similarity score

### Model Components:
- **Base CNN**: Extracts features from signatures
  - Conv2D layers (32, 64 filters)
  - MaxPooling layers
  - Dense layer (128 units)
- **Distance Layer**: Computes Euclidean distance between feature vectors
- **Output**: Single value representing distance (0 = identical, higher = different)

## 🚀 How to Test the Model

### 1. Run Comprehensive Tests

```bash
python test_model.py
```

This will:
- Load the trained model
- Test genuine vs genuine signatures
- Test genuine vs forged signatures
- Show multiple comparisons
- Display accuracy metrics

### 2. Test a Single Pair

```bash
python verify_single.py <reference_signature> <test_signature>
```

**Example:**
```bash
python verify_single.py dataset/person1/genuine/1.jpg dataset/person1/genuine/2.jpg
```

## 📊 Understanding the Results

### Key Metrics:

1. **Distance** (0 to ~1.5)
   - Lower = More similar
   - 0 = Identical
   - >1.0 = Very different

2. **Similarity** (0% to 100%)
   - Calculated as: `(1 - distance) × 100`
   - Higher = More similar
   - 100% = Identical

3. **Verdict**
   - **GENUINE**: Similarity ≥ 70%
   - **FORGED**: Similarity < 70%

### Example Output:

```
🔍 Testing Signature Pair
============================================================
Reference: genuine_1.jpg
Test:      genuine_2.jpg
Expected:  GENUINE
------------------------------------------------------------
📊 Results:
   Distance:   0.3890
   Similarity: 61.10%
------------------------------------------------------------
🎯 Verdict: ✅ GENUINE (Confidence: 61.10%)
```

## 🔍 Verification Logic Explained

### Step-by-Step Process:

#### 1. **Load the Model**
```python
from models.siamese_model import build_siamese_model

# Rebuild architecture
model = build_siamese_model()

# Load trained weights
model.load_weights('models/siamese_trained.h5')
```

#### 2. **Preprocess Images**
```python
def preprocess_signature(image_path):
    # Load in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 128x128
    img = cv2.resize(img, (128, 128))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension: (128, 128) -> (128, 128, 1)
    img = np.expand_dims(img, axis=-1)
    
    return img
```

#### 3. **Compute Similarity**
```python
def compute_similarity(model, img1_path, img2_path):
    # Preprocess both images
    img1 = preprocess_signature(img1_path)
    img2 = preprocess_signature(img2_path)
    
    # Add batch dimension: (128, 128, 1) -> (1, 128, 128, 1)
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Get model prediction (Euclidean distance)
    distance = model.predict([img1_batch, img2_batch])[0][0]
    
    # Convert to similarity percentage
    similarity = max(0, min(100, (1 - distance) * 100))
    
    return distance, similarity
```

#### 4. **Make Decision**
```python
THRESHOLD = 70.0  # Adjustable threshold

if similarity >= THRESHOLD:
    verdict = "GENUINE"
    confidence = similarity
else:
    verdict = "FORGED"
    confidence = 100 - similarity
```

## ⚙️ Configuration

### Adjustable Parameters:

1. **Threshold** (in `test_model.py` or `verify_single.py`)
   ```python
   THRESHOLD = 70.0  # Default: 70%
   ```
   - Higher threshold = Stricter verification
   - Lower threshold = More lenient verification

2. **Image Size** (in `utils/config.py`)
   ```python
   IMG_HEIGHT = 128
   IMG_WIDTH = 128
   ```

## 📈 Expected Performance

### Good Model Performance:
- **Genuine vs Genuine**: Similarity > 70%
- **Genuine vs Forged**: Similarity < 50%

### Current Model Results:
⚠️ **Note**: The current model shows inverted results:
- Genuine pairs: ~48% similarity
- Forged pairs: ~67% similarity

This suggests the model may need:
1. More training epochs
2. Better data augmentation
3. Label verification
4. Hyperparameter tuning

## 🛠️ Troubleshooting

### Issue: Model shows low similarity for genuine signatures

**Solutions:**
1. Retrain with more epochs
2. Check if labels are correct
3. Ensure preprocessing is consistent
4. Try different threshold values

### Issue: Lambda layer loading error

**Solution:** Use the weight loading approach:
```python
model = build_siamese_model()
model.load_weights(SIAMESE_MODEL_SAVE_PATH)
```

## 📝 Code Examples

### Example 1: Verify Two Signatures
```python
from verify_single import verify_signature

result = verify_signature(
    'dataset/person1/genuine/1.jpg',
    'dataset/person1/genuine/2.jpg',
    threshold=70.0
)

print(f"Similarity: {result['similarity']:.2f}%")
print(f"Verdict: {'GENUINE' if result['is_genuine'] else 'FORGED'}")
```

### Example 2: Batch Verification
```python
import os
from test_model import preprocess_signature, compute_similarity, load_trained_model

# Load model once
model = load_trained_model()

# Reference signature
reference = 'dataset/person1/genuine/1.jpg'

# Test multiple signatures
test_folder = 'dataset/person1/genuine'
for filename in os.listdir(test_folder):
    if filename.endswith('.jpg'):
        test_path = os.path.join(test_folder, filename)
        distance, similarity, is_genuine = compute_similarity(
            model, reference, test_path
        )
        print(f"{filename}: {similarity:.2f}% - {'✅' if is_genuine else '❌'}")
```

## 🎯 Next Steps

1. **Improve Model**: Retrain with more epochs and better hyperparameters
2. **Optimize Threshold**: Experiment with different threshold values
3. **Add Preprocessing**: Enhance image preprocessing pipeline
4. **Create API**: Build a REST API for signature verification
5. **Deploy**: Create a web interface for easy testing

## 📚 Files Overview

- `test_model.py` - Comprehensive testing script
- `verify_single.py` - Single pair verification script
- `models/siamese_model.py` - Model architecture
- `models/siamese_trained.h5` - Trained model weights
- `utils/config.py` - Configuration parameters

## 🔗 References

- Siamese Networks: Learning similarity metrics
- Euclidean Distance: Measure of similarity in feature space
- Contrastive Loss: Training objective for Siamese networks
