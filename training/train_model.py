from models.siamese_model import build_siamese_model, save_model
from training.pair_generator import generate_pairs_all
from utils.config import SIAMESE_MODEL_SAVE_PATH, TRAINING_EPOCHS, BATCH_SIZE


def train():
    print("Creating training pairs...")
    X1, X2, Y = generate_pairs_all()

    print("Pairs count:", len(Y))
    if len(Y) == 0:
        raise ValueError("No training pairs found. Check dataset structure.")

    print("Building Siamese model...")
    model = build_siamese_model()

    print("Starting training...")
    model.fit(
        [X1, X2],
        Y,
        epochs=TRAINING_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    save_model(model, SIAMESE_MODEL_SAVE_PATH)
    print("Training completed.")
    print("Model saved at:", SIAMESE_MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
