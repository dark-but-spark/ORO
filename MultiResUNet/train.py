import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
import torch
import argparse
from datetime import datetime

# Import the MultiResUNet model and utility functions
from pytorch.MultiResUNet import MultiResUnet, dice_coef, jacard, saveModel, evaluateModel, trainStep
from dataloading import load_data, split_data

# Define paths for data
IMAGE_DIR = 'data/imgs/'
MASK_DIR = 'data/masks/'

def parse_args():
    """
    Parse command line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train MultiResUNet for image segmentation")
    
    # Data loading arguments
    parser.add_argument('--data-limit', type=int, default=None, 
                        help='Number of samples to load for training (default: None). Use small values for quick testing.')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Proportion of data used for validation (default: 0.1)')
    
    # Image resizing arguments
    parser.add_argument('--scale', action='store_true',
                        help='Enable image scaling')
    parser.add_argument('--scale-factor', type=float, default=0.5,
                        help='Scale factor for images (default: 0.5). E.g., 0.5 reduces to 50%, 1.5 increases to 150%')
    
    # Model arguments
    parser.add_argument('--input-channels', type=int, default=3,
                        help='Number of input image channels (default: 3)')
    parser.add_argument('--output-channels', type=int, default=4,
                        help='Number of output segmentation channels (default: 4)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    
    # Optimization arguments
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0). Set to 0 to disable.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay (L2 regularization) for optimizer (default: 0)')
    
    # Data loading optimization
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading (default: 4). Increase to utilize more CPU cores')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Number of batches loaded in advance by each worker (default: 2)')
    
    # Logging and saving
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging during training')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoints during training')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save model checkpoints (default: models)')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--log-dir', type=str, default='runs/logs',
                        help='Directory for TensorBoard logs (default: runs/logs)')
    
    # Debugging options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    parser.add_argument('--check-data', action='store_true',
                        help='Run data validation checks before training')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda)')
    
    args = parser.parse_args()
    return args

# Main training function
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("MultiResUNet Training Configuration")
    print("=" * 60)
    print(f"Data Limit: {args.data_limit} samples")
    print(f"Validation Split: {args.validation_split:.1%}")
    print(f"Scale Enabled: {args.scale}")
    if args.scale:
        print(f"Scale Factor: {args.scale_factor} ({args.scale_factor*100:.0f}%)")
    print(f"Input Channels: {args.input_channels}")
    print(f"Output Channels: {args.output_channels}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gradient Clipping: {args.gradient_clip}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Device: {args.device}")
    print(f"Debug Mode: {args.debug}")
    print(f"Data Validation: {args.check_data}")
    print(f"TensorBoard: {args.tensorboard}")
    if args.tensorboard:
        print(f"Log Directory: {args.log_dir}")
    print("=" * 60)
    
    # Check for GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup TensorBoard logging if enabled
    if args.tensorboard:
        os.makedirs(args.log_dir, exist_ok=True)
        log_dir = os.path.join(args.log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"\nTensorBoard logs will be saved to: {log_dir}")
    else:
        log_dir = None

    # Load data using memory-efficient approach
    print(f"\nLoading data...")
    
    # Option 1: Use new memory-efficient Dataset approach (recommended for large datasets)
    if args.data_limit is None or args.data_limit > 500:
        print("Using memory-efficient dataset loading (recommended for large datasets)...")
        from dataloading import create_datasets
        
        # Create datasets that will load data on-demand
        train_dataset, val_dataset, n_train, n_val = create_datasets(
            img_dir='data/imgs',
            mask_dir='data/masks',
            limit=args.data_limit,
            train_ratio=1.0 - args.validation_split,
            scale=args.scale,
            scale_factor=args.scale_factor
        )
        
        # Create DataLoaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        
        print(f"✓ Training samples: {n_train}")
        print(f"✓ Validation samples: {n_val}")
        print(f"✓ Memory usage: Minimal (data loaded batch-by-batch)")
        
        # Initialize model BEFORE training
        print(f"\nInitializing model...")
        model = MultiResUnet(
            input_channels=args.input_channels, 
            num_classes=args.output_channels
        ).to(device)
        
        print(f"Model architecture: MultiResUNet")
        print(f"  Input: {args.input_channels} channels")
        print(f"  Output: {args.output_channels} channels")
        
        # Train the model with DataLoaders
        print(f"\nStarting training...")
        print("-" * 60)
        
        history = trainStep(
            model, 
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            device=device,
            learning_rate=args.learning_rate,
            gradient_clip=args.gradient_clip,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            save_model=args.save_model,
            save_dir=args.save_dir,
            verbose=args.verbose,
            log_dir=log_dir  # Pass TensorBoard log directory
        )
    
    else:
        # Option 2: Use traditional approach for small datasets (< 500 samples)
        print("Using traditional in-memory loading (suitable for small datasets)...")
        X, Y = load_data(limit=args.data_limit, 
                        scale=args.scale,
                        scale_factor=args.scale_factor)

        # Validate mask channels BEFORE any processing
        if args.debug or args.check_data:
            print(f"\nData Validation:")
            print(f"  Original Y shape: {Y.shape}")
            print(f"  Y sample unique values: {np.unique(Y[0])}")
            print(f"  Y value range: [{Y.min():.4f}, {Y.max():.4f}]")
            print(f"  Y positive pixel ratio: {Y.sum() / Y.size:.4f}")
        
        # Ensure Y has correct number of channels
        if Y.shape[-1] == 1:
            print("⚠ WARNING: Single channel mask detected. Duplicating to match output channels...")
            Y = np.concatenate([Y] * args.output_channels, axis=-1)
            print(f"After duplication Y shape: {Y.shape}")
        elif Y.shape[-1] != args.output_channels:
            raise ValueError(f"Expected {args.output_channels} channels in mask, got {Y.shape[-1]}")

        # Split data into training and validation sets
        print(f"\nSplitting data (validation={args.validation_split:.1%})...")
        X_train, X_val, Y_train, Y_val = split_data(X, Y, validation=args.validation_split)

        # Define the model
        print(f"\nInitializing model...")
        model = MultiResUnet(
            input_channels=args.input_channels, 
            num_classes=args.output_channels
        ).to(device)
        
        print(f"Model architecture: MultiResUNet")
        print(f"  Input: {args.input_channels} channels")
        print(f"  Output: {args.output_channels} channels")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Print data statistics
        if args.debug:
            print(f"\nData Statistics:")
            print(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
            print(f"  Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
            print(f"  Y_train positive ratio: {Y_train.sum() / Y_train.size:.4f}")

        # Train the model
        print(f"\nStarting training...")
        print("-" * 60)
        
        history = trainStep(
            model, 
            X_train, Y_train, 
            X_val, Y_val, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            device=device,
            learning_rate=args.learning_rate,
            gradient_clip=args.gradient_clip,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            save_model=args.save_model,
            save_dir=args.save_dir,
            verbose=args.verbose,
            log_dir=log_dir  # Pass TensorBoard log directory
        )
    
    print("-" * 60)
    print("Training complete!")
    
    # Save final model if requested
    if args.save_model:
        print(f"\nSaving final model to {args.save_dir}/")
        saveModel(model, args.save_dir)

if __name__ == "__main__":
    main()