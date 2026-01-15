import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import exposure


def extract_texture_features(image):
    """
    Extract texture features from a signature image using Local Binary Pattern (LBP)
    
    Args:
        image (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Texture feature vector
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Local Binary Pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist = hist / (hist.sum() + 1e-7)
    
    return hist


def extract_shape_features(image):
    """
    Extract shape features from a signature image using HOG (Histogram of Oriented Gradients)
    
    Args:
        image (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Shape feature vector
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    return features


def extract_geometric_features(image):
    """
    Extract geometric features from a signature image
    
    Args:
        image (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Geometric feature vector
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold the image to get the signature
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(10)  # Return zeros if no contours found
    
    # Get the largest contour (assuming it's the signature)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate geometric features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h)
    
    # Convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Minimum enclosing circle
    (x_c, y_c), radius = cv2.minEnclosingCircle(largest_contour)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Moments
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments)
    # Flatten the Hu moments
    hu_moments = hu_moments.flatten()
    
    # Combine all geometric features
    features = np.array([
        area, perimeter, aspect_ratio, extent, 
        solidity, circularity, radius
    ])
    
    # Concatenate with Hu moments (first 7, as the last one is noisy)
    geometric_features = np.concatenate([features, hu_moments[:7]])
    
    return geometric_features


def extract_features(image):
    """
    Extract all features from a signature image
    
    Args:
        image (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Combined feature vector
    """
    # Extract different types of features
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(image)
    geometric_features = extract_geometric_features(image)
    
    # Combine all features
    combined_features = np.concatenate([
        texture_features,
        shape_features,
        geometric_features
    ])
    
    return combined_features


def extract_spatial_features(image):
    """
    Extract spatial features from a signature image
    
    Args:
        image (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Spatial feature vector
    """
    # Calculate image statistics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    # Calculate skewness and kurtosis
    from scipy.stats import skew, kurtosis
    flat_image = image.flatten()
    skewness = skew(flat_image)
    kurt = kurtosis(flat_image)
    
    # Calculate energy
    energy = np.sum(image ** 2) / (image.shape[0] * image.shape[1])
    
    # Combine spatial features
    spatial_features = np.array([
        mean_intensity, std_intensity, min_intensity, 
        max_intensity, skewness, kurt, energy
    ])
    
    return spatial_features


if __name__ == "__main__":
    # Test the feature extraction functions
    import sys
    if len(sys.argv) > 1:
        import cv2
        test_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if test_image is not None:
            features = extract_features(test_image)
            print(f"Features extracted successfully. Shape: {features.shape}")
        else:
            print("Could not load image")
    else:
        print("Usage: python feature_extraction.py <image_path>")