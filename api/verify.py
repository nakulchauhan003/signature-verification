from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese_model import build_siamese_model
from utils.config import SIAMESE_MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH, VERIFICATION_THRESHOLD

# Global model
model = None

def load_model():
    global model
    if model is None:
        model = build_siamese_model()
        model.load_weights(SIAMESE_MODEL_SAVE_PATH)
    return model

def preprocess_signature(image_bytes):
    """Preprocess signature from bytes"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            content_type = self.headers['Content-Type']

            # Parse multipart form data
            boundary = content_type.split('boundary=')[1].encode()
            parts = body.split(b'--' + boundary)

            files = {}
            for part in parts:
                if b'filename=' in part:
                    name_start = part.find(b'name="') + 6
                    name_end = part.find(b'"', name_start)
                    name = part[name_start:name_end].decode()

                    data_start = part.find(b'\r\n\r\n') + 4
                    data_end = part.rfind(b'\r\n')
                    files[name] = part[data_start:data_end]

            if 'reference' not in files or 'test' not in files:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Both signatures required'}).encode())
                return

            mdl = load_model()

            img1 = preprocess_signature(files['reference'])
            img2 = preprocess_signature(files['test'])

            img1_batch = np.expand_dims(img1, axis=0)
            img2_batch = np.expand_dims(img2, axis=0)

            distance = mdl.predict([img1_batch, img2_batch], verbose=0)[0][0]
            similarity = max(0, min(100, (1 - distance) * 100))

            threshold = VERIFICATION_THRESHOLD * 100
            is_genuine = similarity >= threshold
            confidence = similarity if is_genuine else (100 - similarity)

            result = {
                'distance': float(distance),
                'similarity': float(similarity),
                'threshold': float(threshold),
                'is_genuine': bool(is_genuine),
                'confidence': float(confidence),
                'verdict': 'GENUINE' if is_genuine else 'FORGED'
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
