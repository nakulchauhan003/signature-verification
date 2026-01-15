from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
import tensorflow.keras.backend as K
from utils.config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


def build_siamese_model():
    """
    Builds and compiles a Siamese Neural Network for signature verification
    """

    # Base CNN (shared)
    def base_network():
        inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        x = Conv2D(32, (3, 3), activation="relu")(inp)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        return Model(inp, x)

    def euclidean_distance(vectors):
        x, y = vectors
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    input_a = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    input_b = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    base = base_network()

    feat_a = base(input_a)
    feat_b = base(input_b)

    distance = Lambda(euclidean_distance)([feat_a, feat_b])

    model = Model([input_a, input_b], distance)
    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


def save_model(model, path):
    model.save(path)
