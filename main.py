import argparse
import os
import sys
from verification.compare import SignatureComparator
from verification.forgery_detection import ForgeryDetector
from verification.drift_check import SignatureDriftDetector
from verification.risk_score import RiskScoreCalculator
from preprocessing.preprocess import preprocess_image
from preprocessing.feature_extraction import extract_features
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Signature Verification System")
    parser.add_argument("--genuine_path", type=str, required=True,
                        help="Path to the genuine signature image")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to the test signature image")
    parser.add_argument("--model_path", type=str, default="models/trained_model.h5",
                        help="Path to the trained model")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for verification decision")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.genuine_path):
        print(f"Error: Genuine signature file not found at {args.genuine_path}")
        sys.exit(1)
    
    if not os.path.exists(args.test_path):
        print(f"Error: Test signature file not found at {args.test_path}")
        sys.exit(1)
    
    # Initialize components
    comparator = SignatureComparator(model_path=args.model_path)
    detector = ForgeryDetector()
    drift_detector = SignatureDriftDetector()
    risk_calculator = RiskScoreCalculator()
    
    try:
        # Perform signature comparison
        print("Performing signature comparison...")
        similarity_score = comparator.compare_signatures(args.genuine_path, args.test_path)
        print(f"Similarity score: {similarity_score:.4f}")
        
        # Perform forgery detection on the test signature
        print("Analyzing test signature for forgery...")
        forgery_analysis = detector.analyze_forgery_type(args.test_path)
        print(f"Forgery score: {forgery_analysis['overall_forgery_score']:.4f}")
        print(f"Is forged: {forgery_analysis['is_forged']}")
        
        # Perform drift detection
        print("Performing drift analysis...")
        genuine_features = extract_features(preprocess_image(args.genuine_path))
        test_features = extract_features(preprocess_image(args.test_path))
        
        drift_analysis = {
            'drift_score': float(np.linalg.norm(genuine_features - test_features)) / 100.0
        }
        print(f"Drift score: {drift_analysis['drift_score']:.4f}")
        
        # Calculate risk score
        print("Calculating risk score...")
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
        
        # Display results
        print("\n" + "="*50)
        print("VERIFICATION RESULTS")
        print("="*50)
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Risk Level: {risk_analysis['risk_level']}")
        print(f"Recommended Action: {risk_analysis['recommended_action']}")
        print(f"Confidence: {risk_analysis['confidence']:.4f}")
        
        print("\nRisk Breakdown:")
        for component, score in risk_analysis['risk_breakdown'].items():
            print(f"  {component}: {score:.4f}")
        
        print("\nComponent Scores:")
        for component, score in risk_analysis['component_scores'].items():
            print(f"  {component}: {score:.4f}")
        
        # Determine final decision
        is_valid = risk_analysis['recommended_action'] in ['ACCEPT', 'REVIEW']
        print(f"\nSignature is: {'VALID' if is_valid else 'INVALID'}")
        
        if is_valid:
            print("✓ Signature verification passed")
        else:
            print("✗ Signature verification failed")
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()