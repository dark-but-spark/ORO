#!/usr/bin/env python3
"""
Debug script to check data loading and model configuration
"""
import argparse
import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_data(args):
    """Check data loading and preprocessing"""
    print("=" * 50)
    print("DATA LOADING CHECK")
    print("=" * 50)
    
    # Check directories
    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    
    print(f"Image directory: {img_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Image files: {len(list(img_dir.glob('*')))}")
    print(f"Mask files: {len(list(mask_dir.glob('*')))}")
    
    # Try to load dataset
    try:
        dataset = CarvanaDataset(img_dir, mask_dir, args.scale)
    except Exception as e:
        print(f"CarvanaDataset failed: {e}")
        try:
            dataset = BasicDataset(img_dir, mask_dir, args.scale)
        except Exception as e2:
            print(f"BasicDataset also failed: {e2}")
            return False
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check first few samples
    print("\nSample checks:")
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            img = sample['image']
            mask = sample['mask']
            
            print(f"Sample {i}:")
            print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
            
            if hasattr(dataset, 'mask_values'):
                print(f"  Unique mask values: {dataset.mask_values}")
            if hasattr(dataset, 'mask_channels'):
                print(f"  Detected mask channels: {dataset.mask_channels}")
                
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
    
    return True

def check_model_config(args):
    """Check model configuration"""
    print("\n" + "=" * 50)
    print("MODEL CONFIGURATION CHECK")
    print("=" * 50)
    
    # Auto-detect mask channels
    detected = None
    if args.mask_channels is None:
        try:
            from train import detect_mask_channels
            detected = detect_mask_channels(Path(args.mask_dir))
        except Exception as e:
            print(f"Auto-detection failed: {e}")
    
    if args.mask_channels is None and detected is not None:
        n_classes = detected
        print(f"Auto-detected mask channels: {detected}")
    else:
        n_classes = args.mask_channels if args.mask_channels is not None else args.classes
        print(f"Using specified classes: {n_classes}")
    
    # Create model
    try:
        model = UNet(n_channels=3, n_classes=n_classes, bilinear=args.bilinear)
        print(f"Model created successfully:")
        print(f"  Input channels: {model.n_channels}")
        print(f"  Output channels: {model.n_classes}")
        print(f"  Bilinear upsampling: {args.bilinear}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  Forward pass successful, output shape: {output.shape}")
            
        return True
    except Exception as e:
        print(f"Model creation failed: {e}")
        return False

def check_training_setup(args):
    """Check training setup"""
    print("\n" + "=" * 50)
    print("TRAINING SETUP CHECK")
    print("=" * 50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check learning rate and batch size
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    
    # Suggest improvements based on common issues
    suggestions = []
    
    if args.learning_rate > 1e-3:
        suggestions.append("Learning rate might be too high, try 1e-4 or 1e-5")
    elif args.learning_rate < 1e-6:
        suggestions.append("Learning rate might be too low, try 1e-4 or 1e-3")
        
    if args.batch_size == 1:
        suggestions.append("Batch size of 1 may cause unstable training, try batch_size=2 or 4 if memory allows")
        
    if args.epochs < 10:
        suggestions.append("Consider increasing epochs for better convergence")
        
    if suggestions:
        print("\nSuggestions for improvement:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("\nNo obvious configuration issues detected")

def main():
    parser = argparse.ArgumentParser(description='Debug UNet training setup')
    parser.add_argument('--img-dir', type=str, default='./data/imgs/', help='Image directory')
    parser.add_argument('--mask-dir', type=str, default='./data/masks/', help='Mask directory')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scaling factor')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--mask-channels', type=int, default=None, help='Mask channels override')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    
    args = parser.parse_args()
    
    success = True
    success &= check_data(args)
    success &= check_model_config(args)
    check_training_setup(args)
    
    print("\n" + "=" * 50)
    if success:
        print("All checks passed! Ready to train.")
        print("Try running: python train.py --epochs 20 --batch-size 2 --learning-rate 1e-4")
    else:
        print("Some checks failed. Please fix the issues above before training.")
    print("=" * 50)

if __name__ == '__main__':
    main()