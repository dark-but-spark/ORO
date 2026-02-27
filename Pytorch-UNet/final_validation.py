import torch
from utils.data_loading import BasicDataset
import numpy as np

def final_mask_validation():
    """最终mask形状验证"""
    print("=== 最终Mask形状验证 ===\n")
    
    # 测试原始尺寸
    print("测试原始尺寸 (scale=1.0):")
    try:
        dataset = BasicDataset('./data/imgs/', './data/masks/', scale=1.0)
        sample = dataset[0]
        
        print(f"  ✓ Image shape: {sample['image'].shape}")
        print(f"  ✓ Mask shape: {sample['mask'].shape}")
        print(f"  ✓ Mask dtype: {sample['mask'].dtype}")
        print(f"  ✓ Mask unique values: {torch.unique(sample['mask'])}")
        print(f"  ✓ Mask range: {sample['mask'].min():.3f} to {sample['mask'].max():.3f}")
        
        # 验证形状合理性
        mask_shape = sample['mask'].shape
        if len(mask_shape) == 2:
            print(f"  ✓ 单通道mask格式正确: {mask_shape}")
        elif len(mask_shape) == 3 and mask_shape[0] == 4:
            print(f"  ✓ 4通道mask格式正确: {mask_shape}")
        else:
            print(f"  ✗ 异常的mask形状: {mask_shape}")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    print("\n=== 数据统计 ===")
    # 统计不同类型mask的数量
    try:
        dataset = BasicDataset('./data/imgs/', './data/masks/', scale=1.0)
        single_channel = 0
        multi_channel = 0
        
        for i in range(min(100, len(dataset))):  # 检查前100个样本
            sample = dataset[i]
            if len(sample['mask'].shape) == 2:
                single_channel += 1
            elif len(sample['mask'].shape) == 3:
                multi_channel += 1
                
        print(f"  前100个样本中:")
        print(f"  - 单通道mask: {single_channel}")
        print(f"  - 多通道mask: {multi_channel}")
        
    except Exception as e:
        print(f"  统计错误: {e}")

if __name__ == "__main__":
    final_mask_validation()