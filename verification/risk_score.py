import numpy as np


class RiskScoreCalculator:
    def __init__(self):
        """
        Initialize the risk score calculator
        """
        self.weights = {
            'similarity_score': 0.4,      # Output from Siamese network
            'forgery_score': 0.3,         # Score from forgery detection
            'drift_score': 0.2,           # Score from drift detection
            'quality_score': 0.1          # Quality of the signature image
        }
    
    def calculate_risk_score(self, similarity_score, forgery_analysis, drift_analysis, quality_metrics=None):
        """
        Calculate the overall risk score for a signature verification
        
        Args:
            similarity_score (float): Similarity score from Siamese network (0-1, higher = more similar)
            forgery_analysis (dict): Results from forgery detection
            drift_analysis (dict): Results from drift detection
            quality_metrics (dict, optional): Quality metrics of the signature
        
        Returns:
            dict: Comprehensive risk analysis
        """
        # Validate inputs
        similarity_score = max(0.0, min(1.0, similarity_score))
        
        # Extract forgery score (0-1, higher = more likely to be forged)
        forgery_score = min(1.0, max(0.0, forgery_analysis.get('overall_forgery_score', 0.0)))
        
        # Extract drift score (0-1, higher = more drift)
        drift_score = drift_analysis.get('drift_score', 0.0) if drift_analysis else 0.0
        drift_score = min(1.0, max(0.0, drift_score))
        
        # Calculate quality score (0-1, higher = better quality)
        if quality_metrics:
            sharpness_score = min(1.0, max(0.0, quality_metrics.get('sharpness', 50) / 100.0))
            contrast_score = min(1.0, max(0.0, quality_metrics.get('contrast', 30) / 50.0))
            quality_score = (sharpness_score + contrast_score) / 2.0
        else:
            quality_score = 1.0  # Default to high quality if not provided
        
        # Adjust similarity score based on quality (low quality signatures should have less weight)
        adjusted_similarity = similarity_score * quality_score
        
        # Calculate risk components
        # Lower similarity = higher risk
        similarity_risk = 1.0 - adjusted_similarity
        
        # Higher forgery score = higher risk
        forgery_risk = forgery_score
        
        # Higher drift score = higher risk
        drift_risk = drift_score
        
        # Lower quality = higher risk
        quality_risk = 1.0 - quality_score
        
        # Calculate weighted risk score
        total_risk = (
            similarity_risk * self.weights['similarity_score'] +
            forgery_risk * self.weights['forgery_score'] +
            drift_risk * self.weights['drift_score'] +
            quality_risk * self.weights['quality_score']
        )
        
        # Determine risk level
        if total_risk < 0.3:
            risk_level = "LOW"
            action = "ACCEPT"
        elif total_risk < 0.6:
            risk_level = "MEDIUM"
            action = "REVIEW"
        else:
            risk_level = "HIGH"
            action = "REJECT"
        
        # Calculate confidence score (opposite of risk, but also considering how clear the decision is)
        confidence = 1.0 - abs(total_risk - 0.5) * 2  # More confident when not near the 0.5 threshold
        
        return {
            'total_risk_score': round(float(total_risk), 4),
            'risk_level': risk_level,
            'recommended_action': action,
            'confidence': round(float(confidence), 4),
            'risk_breakdown': {
                'similarity_risk': round(float(similarity_risk), 4),
                'forgery_risk': round(float(forgery_risk), 4),
                'drift_risk': round(float(drift_risk), 4),
                'quality_risk': round(float(quality_risk), 4)
            },
            'component_scores': {
                'similarity_score': round(float(similarity_score), 4),
                'adjusted_similarity': round(float(adjusted_similarity), 4),
                'forgery_score': round(float(forgery_score), 4),
                'drift_score': round(float(drift_score), 4),
                'quality_score': round(float(quality_score), 4)
            }
        }
    
    def calculate_verification_score(self, risk_analysis):
        """
        Calculate a simple verification score from risk analysis
        
        Args:
            risk_analysis (dict): Risk analysis result from calculate_risk_score
        
        Returns:
            float: Verification score (0-1, higher = more likely to be genuine)
        """
        return 1.0 - risk_analysis['total_risk_score']
    
    def set_weights(self, new_weights):
        """
        Update the weights used in risk calculation
        
        Args:
            new_weights (dict): New weight values for risk components
        """
        # Validate that weights sum to approximately 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, but sum is {total_weight}")
        
        self.weights = new_weights


def normalize_score(score, min_val=0, max_val=1):
    """
    Normalize a score to the range [0, 1]
    
    Args:
        score (float): Input score
        min_val (float): Minimum possible value of input score
        max_val (float): Maximum possible value of input score
    
    Returns:
        float: Normalized score in range [0, 1]
    """
    if max_val == min_val:
        return 0.5  # Return middle value if range is invalid
    
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


if __name__ == "__main__":
    # Example usage
    calculator = RiskScoreCalculator()
    
    # Mock data for demonstration
    similarity = 0.7  # 70% similarity
    forgery_analysis = {
        'overall_forgery_score': 0.2,  # 20% chance of forgery
        'is_forged': False
    }
    drift_analysis = {
        'drift_score': 0.1  # 10% drift
    }
    quality_metrics = {
        'sharpness': 75,
        'contrast': 40
    }
    
    result = calculator.calculate_risk_score(
        similarity_score=similarity,
        forgery_analysis=forgery_analysis,
        drift_analysis=drift_analysis,
        quality_metrics=quality_metrics
    )
    
    print("Risk Analysis Result:")
    print(f"Total Risk Score: {result['total_risk_score']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Verification Score: {calculator.calculate_verification_score(result)}")