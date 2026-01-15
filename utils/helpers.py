import os
import logging
import json
import hashlib
from datetime import datetime
import numpy as np
from PIL import Image


def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Set up logging configuration
    
    Args:
        log_file (str, optional): Path to log file. If None, logs to console
        log_level: Logging level (default: INFO)
    """
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
            logging.StreamHandler()  # Also log to console
        ]
    )


def calculate_file_hash(file_path):
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_image_file(file_path, allowed_extensions=None):
    """
    Validate if a file is a valid image
    
    Args:
        file_path (str): Path to the file
        allowed_extensions (set, optional): Set of allowed extensions
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    
    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        return False
    
    # Try to open the image
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it's a valid image
        return True
    except Exception:
        return False


def save_dict_to_json(data_dict, file_path):
    """
    Save a dictionary to a JSON file
    
    Args:
        data_dict (dict): Dictionary to save
        file_path (str): Path to save the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)


def load_dict_from_json(file_path):
    """
    Load a dictionary from a JSON file
    
    Args:
        file_path (str): Path to the JSON file
    
    Returns:
        dict: Loaded dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_directory_exists(path):
    """
    Ensure that a directory exists, creating it if necessary
    
    Args:
        path (str): Path to the directory
    """
    os.makedirs(path, exist_ok=True)


def get_current_timestamp():
    """
    Get current timestamp as a string
    
    Returns:
        str: Current timestamp in YYYY-MM-DD HH:MM:SS format
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_similarity_percentage(value1, value2, max_value=1.0):
    """
    Calculate similarity percentage between two values
    
    Args:
        value1 (float): First value
        value2 (float): Second value
        max_value (float): Maximum possible value for normalization
    
    Returns:
        float: Similarity percentage (0-100)
    """
    diff = abs(value1 - value2)
    max_diff = max_value
    similarity = max(0, (max_diff - diff) / max_diff) * 100
    return similarity


def normalize_array(arr, min_val=None, max_val=None):
    """
    Normalize an array to the range [0, 1]
    
    Args:
        arr (numpy.ndarray): Input array
        min_val (float, optional): Minimum value for normalization. If None, use array min
        max_val (float, optional): Maximum value for normalization. If None, use array max
    
    Returns:
        numpy.ndarray: Normalized array
    """
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(arr)
    
    return (arr - min_val) / (max_val - min_val)


def get_file_size_mb(file_path):
    """
    Get file size in megabytes
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        float: File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def create_experiment_dir(base_path, experiment_name=None):
    """
    Create a directory for an experiment with timestamp
    
    Args:
        base_path (str): Base directory for experiments
        experiment_name (str, optional): Name of the experiment
    
    Returns:
        str: Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        exp_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_path, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def format_bytes(bytes_value):
    """
    Format bytes value to human readable format
    
    Args:
        bytes_value (int): Size in bytes
    
    Returns:
        str: Human readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


if __name__ == "__main__":
    # Example usage
    print("Helper functions module. Import this module to use its functionality.")