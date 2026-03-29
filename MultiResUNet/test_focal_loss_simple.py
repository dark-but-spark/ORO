"""Test Focal Loss with simple training run"""
import torch
import numpy as np
from pytorch.MultiResUNet import MultiResUnet, trainStep, FocalLoss

if __name__ == '__main__':
    # Create dummy data
    print("Creating dummy dataset...")
    X = np.random.rand(20, 640, 640, 3).astype(np.float32)
    Y = (np.random.rand(20, 640, 640, 4) > 0.8).astype(np.float32)

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")

    # Split
    X_train, X_val = X[:18], X[18:]
    Y_train, Y_val = Y[:18], Y[18:]

    # Create model
    print("\nCreating model...")
    model = MultiResUnet(input_channels=3, num_classes=4)
    print(f"Model created: {model.__class__.__name__}")

    # Test Focal Loss directly
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    x_test = torch.randn(2, 4, 64, 64)
    y_test = torch.randint(0, 2, (2, 4, 64, 64)).float()
    loss = focal_loss(x_test, y_test)
    print(f"Focal Loss test passed! Loss = {loss.item():.4f}")

    # Try training with Focal Loss
    print("\nStarting training with Focal Loss...")
    try:
        history = trainStep(
            model,
            X_train, Y_train, X_val, Y_val,
            epochs=5,
            batch_size=2,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=1e-4,
            num_workers=0,  # Set to 0 for Windows compatibility
            use_focal_loss=True,
            verbose=True
        )
        print("\n[OK] Training completed successfully!")
        print(f"Final loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Dice: {history['val_dice'][-1]:.4f}")
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
