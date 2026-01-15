import numpy as np
import cv2
from datetime import datetime


class SignatureDriftDetector:
    def __init__(self, drift_threshold=0.15):
        """
        Initialize the signature drift detector
        
        Args:
            drift_threshold (float): Threshold for detecting significant signature changes
        """
        self.drift_threshold = drift_threshold
        self.signature_history = {}
    
    def register_signature(self, user_id, signature_features, timestamp=None):
        """
        Register a new signature for a user
        
        Args:
            user_id (str): Unique identifier for the user
            signature_features (numpy.ndarray): Feature vector of the signature
            timestamp (datetime): Timestamp of signature registration (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if user_id not in self.signature_history:
            self.signature_history[user_id] = []
        
        self.signature_history[user_id].append({
            'features': signature_features,
            'timestamp': timestamp
        })
    
    def calculate_signature_drift(self, user_id, new_signature_features):
        """
        Calculate how much a new signature differs from previous signatures of the same user
        
        Args:
            user_id (str): Unique identifier for the user
            new_signature_features (numpy.ndarray): Feature vector of the new signature
        
        Returns:
            dict: Drift analysis results
        """
        if user_id not in self.signature_history or len(self.signature_history[user_id]) == 0:
            return {
                'drift_score': 0.0,
                'is_significant_drift': False,
                'message': 'No previous signatures to compare against'
            }
        
        # Calculate distance to all previous signatures
        distances = []
        for record in self.signature_history[user_id]:
            # Calculate Euclidean distance between feature vectors
            distance = np.linalg.norm(new_signature_features - record['features'])
            distances.append(distance)
        
        # Calculate average distance
        avg_distance = np.mean(distances)
        
        # Calculate drift score (normalized to 0-1 range)
        drift_score = min(avg_distance / 10.0, 1.0)  # Assuming max meaningful distance is around 10
        
        is_significant_drift = drift_score > self.drift_threshold
        
        return {
            'drift_score': float(drift_score),
            'is_significant_drift': is_significant_drift,
            'average_distance': float(avg_distance),
            'num_comparisons': len(distances),
            'message': f'Significant drift detected' if is_significant_drift else f'Normal signature variation'
        }
    
    def update_signature_threshold(self, user_id, new_threshold):
        """
        Update the drift threshold for a specific user
        
        Args:
            user_id (str): Unique identifier for the user
            new_threshold (float): New threshold value
        """
        self.drift_threshold = new_threshold
    
    def get_signature_trend(self, user_id):
        """
        Get the trend of signature changes over time for a user
        
        Args:
            user_id (str): Unique identifier for the user
        
        Returns:
            dict: Trend analysis results
        """
        if user_id not in self.signature_history or len(self.signature_history[user_id]) < 2:
            return {
                'trend': 'insufficient_data',
                'stability_score': 0.0,
                'message': 'Need at least 2 signatures to determine trend'
            }
        
        records = self.signature_history[user_id]
        
        # Calculate distances between consecutive signatures
        consecutive_distances = []
        for i in range(1, len(records)):
            dist = np.linalg.norm(
                records[i]['features'] - records[i-1]['features']
            )
            consecutive_distances.append({
                'distance': dist,
                'timestamp': records[i]['timestamp']
            })
        
        # Calculate trend
        avg_distance = np.mean([d['distance'] for d in consecutive_distances])
        std_distance = np.std([d['distance'] for d in consecutive_distances])
        
        # Stability score (lower std = more stable signature)
        stability_score = max(0.0, 1.0 - (std_distance / (avg_distance + 1e-8)))
        
        if std_distance < avg_distance * 0.3:
            trend = 'stable'
        elif std_distance < avg_distance * 0.6:
            trend = 'moderately_variable'
        else:
            trend = 'highly_variable'
        
        return {
            'trend': trend,
            'stability_score': float(stability_score),
            'average_distance': float(avg_distance),
            'std_distance': float(std_distance),
            'num_signatures': len(records)
        }
    
    def detect_gradual_drift(self, user_id, window_size=3):
        """
        Detect gradual drift in signature patterns over time
        
        Args:
            user_id (str): Unique identifier for the user
            window_size (int): Number of recent signatures to consider
        
        Returns:
            dict: Gradual drift analysis results
        """
        if user_id not in self.signature_history or len(self.signature_history[user_id]) < window_size + 1:
            return {
                'is_gradual_drift': False,
                'drift_rate': 0.0,
                'message': f'Need at least {window_size + 1} signatures to detect gradual drift'
            }
        
        records = self.signature_history[user_id][-window_size-1:]  # Get last window_size+1 records
        
        # Calculate distances between consecutive signatures in the window
        distances = []
        for i in range(1, len(records)):
            dist = np.linalg.norm(
                records[i]['features'] - records[i-1]['features']
            )
            distances.append(dist)
        
        # Calculate drift rate (slope of distances over time)
        time_points = np.array([(r['timestamp'] - records[0]['timestamp']).total_seconds() for r in records[1:]])
        distance_values = np.array(distances)
        
        if len(time_points) > 1 and np.var(time_points) > 0:
            # Calculate linear regression slope
            slope = np.polyfit(time_points, distance_values, 1)[0]
            drift_rate = float(slope)
            is_increasing = drift_rate > 0.001  # Threshold for meaningful increase
        else:
            drift_rate = 0.0
            is_increasing = False
        
        return {
            'drift_rate': drift_rate,
            'is_gradual_drift': is_increasing,
            'message': 'Gradual drift detected' if is_increasing else 'No significant gradual drift'
        }


if __name__ == "__main__":
    import sys
    print("Signature Drift Detector module. Import this module to use its functionality.")