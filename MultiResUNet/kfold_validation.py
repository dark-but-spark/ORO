import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import argparse
from dataloading import create_datasets
from pytorch.MultiResUNet import MultiResUnet, trainStep
from datetime import datetime


def k_fold_cross_validation(k_folds=5, data_limit=None, scale=False, scale_factor=0.5, 
                          epochs=50, batch_size=2, learning_rate=1e-4, device='cuda'):
    """
    Perform k-fold cross validation for MultiResUNet
    
    Args:
        k_folds (int): Number of folds for cross validation
        data_limit (int): Limit the total number of samples to use
        scale (bool): Whether to scale images and masks
        scale_factor (float): Scale factor for images
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        device (str): Device to use for training ('cuda' or 'cpu')
    """
    
    # Get all file paths for dataset
    img_files = sorted(next(os.walk('data/imgs'))[2])
    mask_files = sorted(next(os.walk('data/masks'))[2])
    
    if data_limit:
        img_files = img_files[:data_limit]
        mask_files = mask_files[:data_limit]
    
    print(f"Total samples: {len(img_files)}")
    print(f"Performing {k_folds}-fold cross validation...")
    
    # Initialize metrics storage
    fold_metrics = []
    
    # Define KFold splitter
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_files)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        # Split file lists based on indices from KFold
        train_img_files = [img_files[i] for i in train_idx]
        train_mask_files = [mask_files[i] for i in train_idx]
        val_img_files = [img_files[i] for i in val_idx]
        val_mask_files = [mask_files[i] for i in val_idx]
        
        print(f"Train samples: {len(train_img_files)}, Val samples: {len(val_img_files)}")
        
        # Create custom dataset objects for this fold
        from dataloading import SegmentationDataset
        
        train_dataset = SegmentationDataset(
            img_dir='data/imgs',
            mask_dir='data/masks',
            img_files=train_img_files,
            mask_files=train_mask_files,
            scale=scale,
            scale_factor=scale_factor
        )
        
        val_dataset = SegmentationDataset(
            img_dir='data/imgs',
            mask_dir='data/masks',
            img_files=val_img_files,
            mask_files=val_mask_files,
            scale=scale,
            scale_factor=scale_factor
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Initialize model
        model = MultiResUnet(input_channels=3, num_classes=4).to(device)
        
        # Train the model for this fold
        print(f"Training fold {fold + 1}...")
        
        history = trainStep(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            verbose=False,
            log_dir=f'runs/kfold_fold_{fold+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        # Evaluate model performance on validation set for this fold
        # For simplicity, we'll just store the final validation loss
        # In a real scenario, you'd want to compute more comprehensive metrics
        final_val_loss = history['val_loss'][-1] if 'val_loss' in history else float('inf')
        fold_metrics.append(final_val_loss)
        
        print(f"Fold {fold + 1} completed. Final validation loss: {final_val_loss:.4f}")
        
        # Clean up GPU memory
        del model
        del train_loader
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate and display overall results
    avg_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)
    
    print(f"\n{'='*50}")
    print("CROSS VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Fold metrics: {fold_metrics}")
    print(f"Average validation loss: {avg_metric:.4f} ± {std_metric:.4f}")
    
    return fold_metrics, avg_metric, std_metric


def parse_args():
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation for MultiResUNet")
    
    parser.add_argument('--k-folds', type=int, default=5, 
                        help='Number of folds for cross validation (default: 5)')
    parser.add_argument('--data-limit', type=int, default=None, 
                        help='Number of samples to use (default: None)')
    parser.add_argument('--scale', action='store_true',
                        help='Enable image scaling')
    parser.add_argument('--scale-factor', type=float, default=0.5,
                        help='Scale factor for images (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("K-Fold Cross Validation Configuration")
    print("=" * 60)
    print(f"K-Folds: {args.k_folds}")
    print(f"Data Limit: {args.data_limit}")
    print(f"Scale: {args.scale}")
    if args.scale:
        print(f"Scale Factor: {args.scale_factor}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Run k-fold cross validation
    fold_metrics, avg_metric, std_metric = k_fold_cross_validation(
        k_folds=args.k_folds,
        data_limit=args.data_limit,
        scale=args.scale,
        scale_factor=args.scale_factor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    print(f"\nFinal Results:")
    print(f"Average validation loss: {avg_metric:.4f} ± {std_metric:.4f}")