import torch
import torch.nn as nn
import numpy as np
from utils.data_loading import BasicDataset
from unet import UNet

def test_dimension_consistency():
    """Test that tensor dimensions are handled consistently"""
    print("=== Dimension Consistency Test ===\n")
    
    # Test data loading
    print("1. Testing data loading dimensions:")
    try:
        dataset = BasicDataset('./data/imgs/', './data/masks/', scale=0.5)
        sample = dataset[0]
        
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Mask dtype: {sample['mask'].dtype}")
        
        # Verify 4-channel format
        if len(sample['mask'].shape) == 3:
            channels, height, width = sample['mask'].shape
            if channels == 4:
                print(f"   ✓ 4-channel mask format correct: ({channels}, {height}, {width})")
            else:
                print(f"   ✗ Unexpected channel count: {channels}")
        else:
            print(f"   ✗ Unexpected dimension count: {len(sample['mask'].shape)}")
            
    except Exception as e:
        print(f"   ✗ Data loading error: {e}")
        return False
    
    # Test model forward pass
    print("\n2. Testing model forward pass:")
    try:
        model = UNet(n_channels=3, n_classes=4)
        # Convert sample to proper tensor format with correct dtype
        if hasattr(sample['image'], 'unsqueeze'):
            dummy_input = sample['image'].unsqueeze(0).float()
        else:
            dummy_input = torch.from_numpy(sample['image']).unsqueeze(0).float()
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Input dtype: {dummy_input.dtype}")
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   Output shape: {output.shape}")
            print(f"   Output dtype: {output.dtype}")
            
            # Verify output matches expected dimensions
            expected_shape = (1, 4, dummy_input.shape[2], dummy_input.shape[3])
            if output.shape == expected_shape:
                print(f"   ✓ Model output shape correct: {output.shape}")
            else:
                print(f"   ✗ Unexpected output shape: {output.shape}, expected: {expected_shape}")
                
    except Exception as e:
        print(f"   ✗ Model forward error: {e}")
        return False
    
    # Test loss computation with corrected reshaping
    print("\n3. Testing loss computation with corrected reshaping:")
    try:
        # Convert masks to proper tensor format with correct dtype
        if hasattr(sample['mask'], 'unsqueeze'):
            true_masks = sample['mask'].unsqueeze(0).float()
        else:
            true_masks = torch.from_numpy(sample['mask']).unsqueeze(0).float()
        
        masks_pred = output  # [1, 4, H, W]
        
        print(f"   True masks shape: {true_masks.shape}")
        print(f"   Pred masks shape: {masks_pred.shape}")
        
        # Apply the corrected reshape logic
        batch_size = true_masks.shape[0]
        channels = true_masks.shape[1]
        target_h, target_w = true_masks.shape[2], true_masks.shape[3]
        pred_h, pred_w = masks_pred.shape[2], masks_pred.shape[3]
        
        # Interpolate if needed
        if (target_h, target_w) != (pred_h, pred_w):
            import torch.nn.functional as F
            masks_pred = F.interpolate(masks_pred, size=(target_h, target_w), mode='bilinear', align_corners=False)
            print(f"   After interpolation: {masks_pred.shape}")
        
        # Reshape both tensors
        true_masks_flat = true_masks.permute(0, 2, 3, 1).reshape(-1, channels)
        masks_pred_flat = masks_pred.permute(0, 2, 3, 1).reshape(-1, channels)
        
        print(f"   Flattened true masks: {true_masks_flat.shape}")
        print(f"   Flattened pred masks: {masks_pred_flat.shape}")
        
        # Verify dimensions match
        if true_masks_flat.shape == masks_pred_flat.shape:
            print(f"   ✓ Flattened dimensions match: {true_masks_flat.shape}")
            
            # Test loss computation
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(masks_pred_flat, true_masks_flat)
            print(f"   Loss computation successful: {loss.item():.6f}")
            print("   ✓ All dimension tests passed!")
            return True
        else:
            print(f"   ✗ Dimension mismatch: true {true_masks_flat.shape} vs pred {masks_pred_flat.shape}")
            return False
            
    except Exception as e:
        print(f"   ✗ Loss computation error: {e}")
        return False

if __name__ == "__main__":
    success = test_dimension_consistency()
    if success:
        print("\n🎉 All tests passed! Dimension handling is now consistent.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")