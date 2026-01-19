from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
import tensorflow.keras.backend as K
from utils.config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive Loss Function
    
    This loss function creates a clear separation between similar and dissimilar pairs:
    - For similar pairs (y_true=1): Minimize distance (push closer)
    - For dissimilar pairs (y_true=0): Maximize distance up to margin (push farther)
    
    Args:
        y_true: Labels (1 for genuine pairs, 0 for forged pairs)
        y_pred: Predicted distances from the model
        margin: Maximum distance for dissimilar pairs (default: 1.0)
        
    Returns:
        Loss value
        
    Formula:
        L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
        
        Where:
        - Y = 1 for similar pairs (genuine-genuine)
        - Y = 0 for dissimilar pairs (genuine-forged)
        - D = Euclidean distance
    """
    y_true = K.cast(y_true, y_pred.dtype)
    
    # For similar pairs (y_true=1): penalize large distances
    # Loss = 0.5 * distance^2
    similar_loss = y_true * K.square(y_pred)
    
    # For dissimilar pairs (y_true=0): penalize small distances
    # Loss = 0.5 * max(0, margin - distance)^2
    dissimilar_loss = (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
    
    # Combine both losses
    return K.mean(0.5 * (similar_loss + dissimilar_loss))


def build_siamese_model(use_contrastive_loss=True, margin=1.0):
    """
    Builds and compiles a Siamese Neural Network for signature verification
    
    Args:
        use_contrastive_loss: If True, use contrastive loss; if False, use MSE (default: True)
        margin: Margin for contrastive loss (default: 1.0)
        
    Returns:
        Compiled Keras model
    """

    # Base CNN (shared weights)
    def base_network():
        """
        Shared CNN that extracts features from signature images
        """
        inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        
        # First convolutional block
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Second convolutional block
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Flatten and dense layer
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        
        return Model(inp, x)

    def euclidean_distance(vectors):
        """
        Compute Euclidean distance between two feature vectors
        """
        x, y = vectors
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    # Define inputs for the two signature images
    input_a = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    input_b = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Create shared base network
    base = base_network()

    # Get feature vectors for both inputs
    feat_a = base(input_a)
    feat_b = base(input_b)

    # Compute distance between feature vectors
    distance = Lambda(euclidean_distance)([feat_a, feat_b])

    # Create the Siamese model
    model = Model([input_a, input_b], distance)
    
    # Compile with appropriate loss function
    if use_contrastive_loss:
        print("✅ Using Contrastive Loss (margin={})".format(margin))
        model.compile(
            optimizer="adam",
            loss=lambda y_true, y_pred: contrastive_loss(y_true, y_pred, margin=margin),
            metrics=['accuracy']
        )
    else:
        print("⚠️  Using Mean Squared Error Loss")
        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

    return model


def save_model(model, path):
    """Save the trained model"""
    model.save(path)
    print(f"Model saved to: {path}")
