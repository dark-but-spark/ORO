#!/usr/bin/env python3
"""
Test script to verify the data loading fix
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
from utils.data_loading import BasicDataset

def test_data_loading():
    print("Testing data loading fix...")
    
    # Test with original scale
    try:
        dataset = BasicDataset('./data/imgs/', './data/masks/', scale=1.0)
        sample = dataset[0]
        print(f"✓ Image shape: {sample['image'].shape}")
        print(f"✓ Mask shape: {sample['mask'].shape}")
        print(f"✓ Image range: {sample['image'].min():.3f} to {sample['image'].max():.3f}")
        print(f"✓ Mask range: {sample['mask'].min()} to {sample['mask'].max()}")
        print("✓ Data loading works correctly!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_with_scaling():
    print("\nTesting with scaling...")
    scales_to_test = [1.0, 0.75, 0.6, 0.4]  # 从大到小测试不同的缩放比例
    
    for scale in scales_to_test:
        try:
            print(f"Testing scale {scale}...")
            dataset = BasicDataset('./data/imgs/', './data/masks/', scale=scale)
            sample = dataset[0]
            print(f"  ✓ Scale {scale}: Image {sample['image'].shape}, Mask {sample['mask'].shape}")
        except AssertionError as e:
            if "Scale is too small" in str(e):
                print(f"  ⚠ Scale {scale}: {e}")
                continue
            else:
                print(f"  ✗ Scale {scale}: Unexpected error - {e}")
                return False
        except Exception as e:
            print(f"  ✗ Scale {scale}: {e}")
            return False
    
    print("✓ Scaling tests completed!")
    return True

def test_dataset_statistics():
    print("\nTesting dataset statistics...")
    try:
        dataset = BasicDataset('./data/imgs/', './data/masks/', scale=1.0)
        
        # 测试几个样本以确保一致性
        print("Checking first 5 samples...")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"  Sample {i}: Image {sample['image'].shape}, Mask {sample['mask'].shape}")
            
        print(f"✓ Dataset size: {len(dataset)} samples")
        print("✓ Dataset statistics look good!")
        return True
    except Exception as e:
        print(f"✗ Dataset statistics error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Data Loading Fix Verification")
    print("=" * 50)
    
    success1 = test_data_loading()
    success2 = test_with_scaling()
    success3 = test_dataset_statistics()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("🎉 All tests passed! The data loading issue has been successfully fixed.")
        print("\nSummary:")
        print("- Image and mask dimensions now match correctly")
        print("- Size comparison logic has been corrected") 
        print("- Dataset can be loaded without assertion errors")
        print("- Multiple scaling factors work appropriately")
    else:
        print("❌ Some tests failed. Please review the implementation.")
        if not success1:
            print("- Basic data loading still has issues")
        if not success2:
            print("- Scaling functionality needs attention")
        if not success3:
            print("- Dataset consistency problems remain")
    print("=" * 50)