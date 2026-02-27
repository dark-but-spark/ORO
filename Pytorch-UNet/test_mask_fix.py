import torch
from utils.data_loading import BasicDataset
import numpy as np

def test_mask_shapes():
    """测试修复后的mask形状"""
    print("=== 修复后mask形状测试 ===\n")
    
    # 测试不同缩放比例
    scales = [1.0, 0.5]
    
    for scale in scales:
        print(f"测试缩放比例: {scale}")
        try:
            dataset = BasicDataset('./data/imgs/', './data/masks/', scale=scale)
            sample = dataset[0]
            
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Mask shape: {sample['mask'].shape}")
            print(f"  Mask dtype: {sample['mask'].dtype}")
            print(f"  Mask unique values: {torch.unique(sample['mask'])}")
            print(f"  Mask range: {sample['mask'].min():.3f} to {sample['mask'].max():.3f}")
            
            # 验证形状是否合理
            if len(sample['mask'].shape) == 2:
                print("  ✓ 单通道mask格式正确")
            elif len(sample['mask'].shape) == 3:
                channels = sample['mask'].shape[0]
                if channels == 4:
                    print("  ✓ 4通道mask格式正确")
                else:
                    print(f"  ⚠ 意外的通道数: {channels}")
            else:
                print(f"  ✗ 异常的mask维度: {len(sample['mask'].shape)}")
            
            print()
            
        except Exception as e:
            print(f"  ✗ 错误: {e}\n")

def test_different_mask_types():
    """测试不同类型mask文件的处理"""
    print("=== 不同类型mask文件测试 ===\n")
    
    from pathlib import Path
    
    mask_dir = Path('./data/masks/')
    
    # 测试几个不同的文件
    test_files = [
        '102_1_6_44_jpg.rf.008d7a4772e4bf8d621bd815723ae428.npz',  # 单通道
        '102_1_6_44_jpg.rf.008d7a4772e4bf8d621bd815723ae428_4ch.npz',  # 4通道
        '102_1_6_44_jpg.rf.008d7a4772e4bf8d621bd815723ae428.png',  # 单通道PNG
        '102_1_6_44_jpg.rf.008d7a4772e4bf8d621bd815723ae428_4ch.png'  # 4通道PNG
    ]
    
    for filename in test_files:
        file_path = mask_dir / filename
        if file_path.exists():
            print(f"测试文件: {filename}")
            try:
                if filename.endswith('.npz'):
                    import numpy as np
                    data = np.load(file_path)
                    for key in data.keys():
                        arr = data[key]
                        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                        if arr.ndim > 2:
                            print(f"    sample slice: {arr[0].shape if len(arr) > 0 else 'N/A'}")
                else:
                    from PIL import Image
                    img = Image.open(file_path)
                    print(f"  PIL Image: size={img.size}, mode={img.mode}")
                    if hasattr(img, 'n_frames'):
                        print(f"  frames: {img.n_frames}")
                        
            except Exception as e:
                print(f"  错误: {e}")
            print()

if __name__ == "__main__":
    test_mask_shapes()
    test_different_mask_types()