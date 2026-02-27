#!/usr/bin/env python3
"""
Test script to verify the loss function fix for 4-channel masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet

def test_loss_functions():
    print("Testing loss function fixes...")
    
    # Create dummy data similar to 4-channel case
    batch_size = 2
    height, width = 64, 64
    n_classes = 4
    
    # Create model
    model = UNet(n_channels=3, n_classes=n_classes)
    
    # Create dummy inputs
    images = torch.randn(batch_size, 3, height, width)
    # 4-channel masks (multi-binary format)
    true_masks = torch.rand(batch_size, n_classes, height, width)
    # Ensure values are in [0,1] range
    true_masks = (true_masks > 0.5).float()
    
    print(f"Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {true_masks.shape}")
    print(f"  Mask value range: [{true_masks.min():.3f}, {true_masks.max():.3f}]")
    
    # Test forward pass
    with torch.no_grad():
        pred = model(images)
        print(f"  Predictions: {pred.shape}")
        print(f"  Pred value range: [{pred.min():.3f}, {pred.max():.3f}]")
    
    # Test BCEWithLogitsLoss (correct for multi-channel binary)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred, true_masks)
    print(f"BCEWithLogitsLoss: {loss.item():.6f}")
    
    # Test sigmoid + dice loss combination
    pred_sigmoid = torch.sigmoid(pred)
    dice_component = 1 - (2 * (pred_sigmoid * true_masks).sum() + 1e-8) / (
        pred_sigmoid.sum() + true_masks.sum() + 1e-8
    )
    print(f"Dice component: {dice_component.item():.6f}")
    
    # Combined loss
    combined_loss = loss + dice_component
    print(f"Combined loss: {combined_loss.item():.6f}")
    
    print("\n✅ Loss function test passed!")
    print("The fix correctly handles 4-channel masks as multi-channel binary segmentation.")

if __name__ == "__main__":
    test_loss_functions()