import numpy as np
from models.siamese_model import SiameseNetwork
from preprocessing.preprocess import preprocess_image
import os


class SignatureComparator:
    def __init__(self, model_path="models/trained_model.h5"):
        """
        Initialize the signature comparator with a trained model
        
        Args:
            model_path (str): Path to the trained model
        """
        self.siamese = SiameseNetwork()
        self.model = self.siamese.get_model()
        
        # Load the trained model if it exists and is valid
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}. Using untrained model.")
        else:
            print(f"Warning: Model not found at {model_path}. Using untrained model.")
    
    def compare_signatures(self, signature1_path, signature2_path):
        """
        Compare two signature images and return similarity score
        
        Args:
            signature1_path (str): Path to the first signature image
            signature2_path (str): Path to the second signature image
        
        Returns:
            float: Similarity score between 0 and 1 (higher means more similar)
        """
        # Preprocess both signatures
        sig1 = preprocess_image(signature1_path)
        sig2 = preprocess_image(signature2_path)
        
        # Add batch dimension
        sig1 = np.expand_dims(sig1, axis=0)
        sig2 = np.expand_dims(sig2, axis=0)
        
        # Make prediction
        prediction = self.model.predict([sig1, sig2], verbose=0)
        
        # Return the similarity score (probability that signatures are genuine)
        return float(prediction[0][0])
    
    def is_genuine(self, signature1_path, signature2_path, threshold=0.5):
        """
        Determine if two signatures are from the same person based on similarity
        
        Args:
            signature1_path (str): Path to the first signature image
            signature2_path (str): Path to the second signature image
            threshold (float): Threshold for determining genuineness (0-1)
        
        Returns:
            bool: True if signatures are genuine (similar), False otherwise
        """
        similarity_score = self.compare_signatures(signature1_path, signature2_path)
        return similarity_score >= threshold


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        comparator = SignatureComparator()
        score = comparator.compare_signatures(sys.argv[1], sys.argv[2])
        is_genuine = comparator.is_genuine(sys.argv[1], sys.argv[2])
        print(f"Similarity score: {score:.4f}")
        print(f"Is genuine: {is_genuine}")
    else:
        print("Usage: python compare.py <signature1_path> <signature2_path>")