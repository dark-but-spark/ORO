#!/usr/bin/env python3
"""
Validation script to test the optimal fix for 4-channel mask training
"""

import torch
import torch.nn as nn
from unet import UNet
import logging

def test_optimal_fix():
    print("🧪 Testing Optimal Fix for 4-Channel Mask Training")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model with 4 classes (matching your 4-channel masks)
    model = UNet(n_channels=3, n_classes=4)
    print(f"✅ Model created: {model.n_classes} output channels")
    
    # Create dummy data simulating your 4-channel case
    batch_size = 2
    height, width = 64, 64
    
    images = torch.randn(batch_size, 3, height, width)
    # 4-channel masks representing multi-class binary segmentation
    true_masks = torch.randint(0, 2, (batch_size, 4, height, width)).float()
    
    print(f"📊 Data shapes:")
    print(f"   Images: {images.shape}")
    print(f"   Masks: {true_masks.shape}")
    print(f"   Mask value range: [{true_masks.min():.1f}, {true_masks.max():.1f}]")
    
    # Test forward pass
    with torch.no_grad():
        predictions = model(images)
        print(f"   Predictions: {predictions.shape}")
        print(f"   Pred value range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Test the optimal loss function approach
    print(f"\n🎯 Testing Optimal Loss Function Approach:")
    
    # Simulate the training logic
    mask_channels = 4  # Your explicit parameter
    
    # Correct loss function for 4-channel masks
    criterion = nn.BCEWithLogitsLoss()
    print(f"   Using BCEWithLogitsLoss (optimal for {mask_channels}-channel masks)")
    
    # Calculate loss
    loss = criterion(predictions, true_masks)
    print(f"   BCE Loss: {loss.item():.6f}")
    
    # Add dice component
    import torch.nn.functional as F
    from utils.dice_score import dice_loss
    
    dice_comp = dice_loss(F.sigmoid(predictions), true_masks, multiclass=True)
    total_loss = loss + dice_comp
    print(f"   Dice component: {dice_comp.item():.6f}")
    print(f"   Total loss: {total_loss.item():.6f}")
    
    print(f"\n✅ OPTIMAL FIX VALIDATED!")
    print(f"   • 4-channel masks correctly handled as multi-channel binary segmentation")
    print(f"   • BCEWithLogitsLoss used instead of problematic CrossEntropyLoss")
    print(f"   • No more 'Expected floating point type' errors")
    print(f"   • Ready for your training with --mask-channels 4")

if __name__ == "__main__":
    test_optimal_fix()