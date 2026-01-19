"""
Flask Web Application for Signature Verification
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import numpy as np
import cv2
from werkzeug.utils import secure_filename

# Add project root to path
sys.path.insert(0, '.')

from models.siamese_model import build_siamese_model
from utils.config import SIAMESE_MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH, VERIFICATION_THRESHOLD

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        print("Loading model...")
        model = build_siamese_model()
        model.load_weights(SIAMESE_MODEL_SAVE_PATH)
        print("Model loaded!")
    return model

def preprocess_signature(image_path):
    """Preprocess signature image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def verify_signatures(reference_path, test_path):
    """Verify if two signatures match"""
    model = load_model()
    
    # Preprocess
    img1 = preprocess_signature(reference_path)
    img2 = preprocess_signature(test_path)
    
    # Add batch dimension
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Predict
    distance = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    similarity = max(0, min(100, (1 - distance) * 100))
    
    # Determine verdict
    threshold = VERIFICATION_THRESHOLD * 100
    is_genuine = similarity >= threshold
    confidence = similarity if is_genuine else (100 - similarity)
    
    return {
        'distance': float(distance),
        'similarity': float(similarity),
        'threshold': float(threshold),
        'is_genuine': bool(is_genuine),
        'confidence': float(confidence),
        'verdict': 'GENUINE' if is_genuine else 'FORGED'
    }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    """Verification endpoint"""
    try:
        if 'reference' not in request.files or 'test' not in request.files:
            return jsonify({'error': 'Both signatures required'}), 400
        
        ref_file = request.files['reference']
        test_file = request.files['test']
        
        if ref_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Save files
        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                'ref_' + secure_filename(ref_file.filename))
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                 'test_' + secure_filename(test_file.filename))
        
        ref_file.save(ref_path)
        test_file.save(test_path)
        
        # Verify
        result = verify_signatures(ref_path, test_path)
        
        # Cleanup
        os.remove(ref_path)
        os.remove(test_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🚀 SIGNATURE VERIFICATION WEB APP")
    print("="*60)
    print("Server: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)