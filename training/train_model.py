from models.siamese_model import build_siamese_model, save_model
from training.pair_generator import generate_pairs_all
from utils.config import (
    SIAMESE_MODEL_SAVE_PATH, 
    TRAINING_EPOCHS, 
    BATCH_SIZE,
    USE_CONTRASTIVE_LOSS,
    CONTRASTIVE_MARGIN
)


def train():
    """Train the Siamese model with contrastive loss"""
    print("="*60)
    print("🚀 TRAINING SIAMESE NETWORK FOR SIGNATURE VERIFICATION")
    print("="*60)
    
    print("\n📊 Creating training pairs...")
    X1, X2, Y = generate_pairs_all()

    print(f"\n✅ Pairs generated:")
    print(f"   Total pairs: {len(Y)}")
    print(f"   Genuine pairs (Y=1): {sum(Y)}")
    print(f"   Forged pairs (Y=0): {len(Y) - sum(Y)}")
    
    if len(Y) == 0:
        raise ValueError("No training pairs found. Check dataset structure.")

    print(f"\n🏗️  Building Siamese model...")
    print(f"   Contrastive Loss: {'✅ Enabled' if USE_CONTRASTIVE_LOSS else '❌ Disabled (using MSE)'}")
    if USE_CONTRASTIVE_LOSS:
        print(f"   Margin: {CONTRASTIVE_MARGIN}")
    
    model = build_siamese_model(
        use_contrastive_loss=USE_CONTRASTIVE_LOSS,
        margin=CONTRASTIVE_MARGIN
    )

    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: {TRAINING_EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Validation Split: 10%")
    
    print(f"\n{'='*60}")
    print("🏋️  Starting training...")
    print(f"{'='*60}\n")
    
    history = model.fit(
        [X1, X2],
        Y,
        epochs=TRAINING_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    print(f"\n{'='*60}")
    print("💾 Saving model...")
    save_model(model, SIAMESE_MODEL_SAVE_PATH)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"📁 Model saved at: {SIAMESE_MODEL_SAVE_PATH}")
    
    # Display final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\n📊 Final Metrics:")
    print(f"   Training Loss: {final_loss:.4f}")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    
    if USE_CONTRASTIVE_LOSS and 'accuracy' in history.history:
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"   Training Accuracy: {final_acc:.4f}")
        print(f"   Validation Accuracy: {final_val_acc:.4f}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Run: python test_model.py")
    print(f"   2. Verify the model shows:")
    print(f"      • Genuine vs Genuine: HIGH similarity (>70%)")
    print(f"      • Genuine vs Forged: LOW similarity (<30%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
