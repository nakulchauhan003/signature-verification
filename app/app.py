from flask import Flask, request, render_template, jsonify
import os
from verification.compare import SignatureComparator
from verification.forgery_detection import ForgeryDetector
from verification.drift_check import SignatureDriftDetector
from verification.risk_score import RiskScoreCalculator
from preprocessing.preprocess import preprocess_image
from preprocessing.feature_extraction import extract_features
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components
comparator = SignatureComparator()
detector = ForgeryDetector()
drift_detector = SignatureDriftDetector()
risk_calculator = RiskScoreCalculator()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/verify', methods=['POST'])
def verify_signature():
    """Verify a signature against a stored genuine signature"""
    try:
        # Check if files were uploaded
        if 'genuine_signature' not in request.files or 'test_signature' not in request.files:
            return jsonify({'error': 'Both genuine and test signatures are required'}), 400
        
        genuine_file = request.files['genuine_signature']
        test_file = request.files['test_signature']
        
        if genuine_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400
        
        # Save uploaded files
        genuine_path = os.path.join(app.config['UPLOAD_FOLDER'], 'genuine_temp.png')
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_temp.png')
        
        genuine_file.save(genuine_path)
        test_file.save(test_path)
        
        # Perform verification
        similarity_score = comparator.compare_signatures(genuine_path, test_path)
        
        # Perform forgery detection on the test signature
        forgery_analysis = detector.analyze_forgery_type(test_path)
        
        # Extract features for drift detection (using the genuine signature as baseline)
        genuine_features = extract_features(preprocess_image(genuine_path))
        test_features = extract_features(preprocess_image(test_path))
        
        # Calculate drift (this is a simplified approach)
        drift_analysis = {
            'drift_score': float(np.linalg.norm(genuine_features - test_features)) / 100.0
        }
        
        # Calculate risk score
        quality_metrics = {
            'sharpness': 75,  # Placeholder - would come from actual quality assessment
            'contrast': 40    # Placeholder - would come from actual quality assessment
        }
        
        risk_analysis = risk_calculator.calculate_risk_score(
            similarity_score=similarity_score,
            forgery_analysis=forgery_analysis,
            drift_analysis=drift_analysis,
            quality_metrics=quality_metrics
        )
        
        # Determine if signature is valid based on risk analysis
        is_valid = risk_analysis['recommended_action'] in ['ACCEPT', 'REVIEW']
        
        # Prepare response
        result = {
            'is_valid': is_valid,
            'similarity_score': float(similarity_score),
            'risk_analysis': risk_analysis,
            'message': f'Signature verification completed with risk level: {risk_analysis["risk_level"]}'
        }
        
        # Clean up temporary files
        os.remove(genuine_path)
        os.remove(test_path)
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up temporary files if they exist
        temp_files = [
            os.path.join(app.config['UPLOAD_FOLDER'], 'genuine_temp.png'),
            os.path.join(app.config['UPLOAD_FOLDER'], 'test_temp.png')
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return jsonify({'error': str(e)}), 500


@app.route('/batch_verify', methods=['POST'])
def batch_verify():
    """Verify multiple signatures against a single genuine signature"""
    try:
        if 'genuine_signature' not in request.files or 'test_signatures' not in request.files:
            return jsonify({'error': 'Genuine signature and test signatures are required'}), 400
        
        genuine_file = request.files['genuine_signature']
        test_files = request.files.getlist('test_signatures')
        
        if genuine_file.filename == '':
            return jsonify({'error': 'Genuine signature must be selected'}), 400
        
        if len(test_files) == 0:
            return jsonify({'error': 'At least one test signature must be provided'}), 400
        
        # Save genuine signature
        genuine_path = os.path.join(app.config['UPLOAD_FOLDER'], 'genuine_temp.png')
        genuine_file.save(genuine_path)
        
        results = []
        for i, test_file in enumerate(test_files):
            if test_file.filename != '':
                # Save test signature
                test_path = os.path.join(app.config['UPLOAD_FOLDER'], f'test_temp_{i}.png')
                test_file.save(test_path)
                
                # Perform verification
                similarity_score = comparator.compare_signatures(genuine_path, test_path)
                
                # Perform forgery detection
                forgery_analysis = detector.analyze_forgery_type(test_path)
                
                # Calculate drift
                genuine_features = extract_features(preprocess_image(genuine_path))
                test_features = extract_features(preprocess_image(test_path))
                
                drift_analysis = {
                    'drift_score': float(np.linalg.norm(genuine_features - test_features)) / 100.0
                }
                
                # Calculate risk
                quality_metrics = {
                    'sharpness': 75,
                    'contrast': 40
                }
                
                risk_analysis = risk_calculator.calculate_risk_score(
                    similarity_score=similarity_score,
                    forgery_analysis=forgery_analysis,
                    drift_analysis=drift_analysis,
                    quality_metrics=quality_metrics
                )
                
                is_valid = risk_analysis['recommended_action'] in ['ACCEPT', 'REVIEW']
                
                results.append({
                    'file_name': test_file.filename,
                    'is_valid': is_valid,
                    'similarity_score': float(similarity_score),
                    'risk_analysis': risk_analysis
                })
                
                # Clean up test file
                os.remove(test_path)
        
        # Clean up genuine file
        os.remove(genuine_path)
        
        return jsonify({'results': results})
    
    except Exception as e:
        # Clean up any temporary files
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.startswith('genuine_temp') or file.startswith('test_temp'):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)