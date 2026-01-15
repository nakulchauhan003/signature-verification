import cv2
import numpy as np
from preprocessing.preprocess import preprocess_image
from preprocessing.feature_extraction import extract_features


class ForgeryDetector:
    def __init__(self):
        """
        Initialize the forgery detector with heuristic rules
        """
        pass

    def detect_copy_move_forgery(self, image_path):
        """
        Detect copy-move forgery in signature
        
        Args:
            image_path (str): Path to the signature image
        
        Returns:
            dict: Analysis results for copy-move forgery
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Look for repeated patterns that might indicate copy-move forgery
        # This is a simplified approach - in practice, more sophisticated methods would be used
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        diff = cv2.absdiff(image, blur)
        
        # Calculate variance in small regions to detect uniform areas
        # which might indicate copy-move operations
        block_size = 16
        h, w = image.shape
        uniform_regions = 0
        total_blocks = 0
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                variance = np.var(block)
                if variance < 50:  # Threshold for uniformity
                    uniform_regions += 1
                total_blocks += 1
        
        uniformity_ratio = uniform_regions / total_blocks if total_blocks > 0 else 0
        
        return {
            "uniformity_ratio": uniformity_ratio,
            "is_suspicious": uniformity_ratio > 0.3,  # If more than 30% is uniform, it's suspicious
            "score": uniformity_ratio
        }

    def detect_quality_issues(self, image_path):
        """
        Detect quality issues that might indicate forgery
        
        Args:
            image_path (str): Path to the signature image
        
        Returns:
            dict: Analysis results for quality issues
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate image sharpness using Laplacian
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Calculate histogram uniformity (affects quality perception)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_uniformity = cv2.compareHist(hist, np.ones_like(hist) * (len(image) * len(image[0]) / 256.0), cv2.HISTCMP_BHATTACHARYYA)
        
        # Calculate contrast (std deviation of pixel values)
        contrast = np.std(image)
        
        return {
            "sharpness": laplacian_var,
            "histogram_uniformity": hist_uniformity,
            "contrast": contrast,
            "is_low_quality": laplacian_var < 50 or contrast < 20,  # Thresholds are adjustable
            "quality_score": min(laplacian_var / 100.0, 1.0)  # Normalize to 0-1 range
        }

    def detect_geometric_inconsistencies(self, image_path):
        """
        Detect geometric inconsistencies that might indicate forgery
        
        Args:
            image_path (str): Path to the signature image
        
        Returns:
            dict: Analysis results for geometric inconsistencies
        """
        image = preprocess_image(image_path)
        features = extract_features(image)
        
        # This would normally compare geometric features against a baseline
        # For now, we'll just return some geometric measurements
        # In a real implementation, we would have a baseline of genuine signatures
        # to compare against
        
        return {
            "geometric_consistency_score": 0.8,  # Placeholder value
            "is_geometrically_consistent": True,  # Placeholder value
        }

    def detect_edge_artifacts(self, image_path):
        """
        Detect edge artifacts that might indicate forgery
        
        Args:
            image_path (str): Path to the signature image
        
        Returns:
            dict: Analysis results for edge artifacts
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Use Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        
        # Calculate edge continuity (simplified)
        # Count edge segments
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_segments = len(contours)
        
        return {
            "edge_density": edge_density,
            "edge_segments": edge_segments,
            "is_edge_artifact_suspicious": edge_density < 0.05 or edge_density > 0.5,  # Too few or too many edges
            "edge_artifact_score": min(edge_density * 2, 1.0) if edge_density <= 0.5 else min((1.0 - edge_density) * 2, 1.0)
        }

    def analyze_forgery_type(self, image_path):
        """
        Analyze the signature to detect potential forgery types
        
        Args:
            image_path (str): Path to the signature image
        
        Returns:
            dict: Comprehensive forgery analysis results
        """
        copy_move_result = self.detect_copy_move_forgery(image_path)
        quality_result = self.detect_quality_issues(image_path)
        geometric_result = self.detect_geometric_inconsistencies(image_path)
        edge_result = self.detect_edge_artifacts(image_path)
        
        # Calculate overall forgery score
        # This is a simplified scoring mechanism
        forgery_score = (
            copy_move_result["score"] * 0.3 +
            (1 - quality_result["quality_score"]) * 0.25 +
            (1 - geometric_result["geometric_consistency_score"]) * 0.25 +
            (1 - edge_result["edge_artifact_score"]) * 0.2
        )
        
        # Determine forgery type based on results
        forgery_types = []
        if copy_move_result["is_suspicious"]:
            forgery_types.append("copy-move")
        if quality_result["is_low_quality"]:
            forgery_types.append("low-quality")
        if not geometric_result["is_geometrically_consistent"]:
            forgery_types.append("geometric-inconsistency")
        if edge_result["is_edge_artifact_suspicious"]:
            forgery_types.append("edge-artifact")
        
        return {
            "copy_move_analysis": copy_move_result,
            "quality_analysis": quality_result,
            "geometric_analysis": geometric_result,
            "edge_analysis": edge_result,
            "overall_forgery_score": min(forgery_score, 1.0),
            "detected_forgery_types": forgery_types,
            "is_forged": len(forgery_types) > 0 or forgery_score > 0.5
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = ForgeryDetector()
        result = detector.analyze_forgery_type(sys.argv[1])
        print("Forgery Analysis Results:")
        print(f"Overall Forgery Score: {result['overall_forgery_score']:.4f}")
        print(f"Is Forged: {result['is_forged']}")
        print(f"Detected Forgery Types: {result['detected_forgery_types']}")
    else:
        print("Usage: python forgery_detection.py <image_path>")