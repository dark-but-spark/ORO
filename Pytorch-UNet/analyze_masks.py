import numpy as np
from pathlib import Path
from PIL import Image
import torch
from utils.data_loading import BasicDataset

def analyze_mask_structure():
    """详细分析mask数据结构"""
    print("=== 详细Mask结构分析 ===\n")
    
    # 检查数据加载器处理后的结果
    print("1. 数据加载器输出:")
    try:
        dataset = BasicDataset('./data/imgs/', './data/masks/', scale=1.0)
        sample = dataset[0]
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Mask dtype: {sample['mask'].dtype}")
        print(f"   Mask unique values: {torch.unique(sample['mask'])}")
        print(f"   Mask min/max: {sample['mask'].min():.3f} / {sample['mask'].max():.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. 原始文件结构分析:")
    mask_dir = Path('./data/masks/')
    
    # 分析NPZ文件
    npz_files = list(mask_dir.glob('*.npz'))
    if npz_files:
        print("   NPZ文件:")
        for npz_file in npz_files[:2]:
            try:
                data = np.load(npz_file)
                keys = list(data.keys())
                print(f"     {npz_file.name}:")
                for key in keys:
                    arr = data[key]
                    print(f"       {key}: shape={arr.shape}, dtype={arr.dtype}")
                    if arr.ndim > 2:
                        print(f"         sample slice shape: {arr[0].shape if len(arr) > 0 else 'N/A'}")
                    print(f"         unique values: {np.unique(arr)[:10]}")
            except Exception as e:
                print(f"     Error loading {npz_file}: {e}")
    
    # 分析PNG文件
    png_files = list(mask_dir.glob('*.png'))
    if png_files:
        print("   PNG文件:")
        for png_file in png_files[:2]:
            try:
                img = Image.open(png_file)
                print(f"     {png_file.name}:")
                print(f"       size: {img.size}")
                print(f"       mode: {img.mode}")
                print(f"       bands: {img.getbands()}")
                
                # 转换为numpy检查
                np_img = np.array(img)
                print(f"       numpy shape: {np_img.shape}")
                print(f"       unique values: {np.unique(np_img)[:10]}")
            except Exception as e:
                print(f"     Error loading {png_file}: {e}")

    print("\n3. 四维数据检查:")
    # 特别检查4通道相关文件
    ch4_files = list(mask_dir.glob('*_4ch*'))
    if ch4_files:
        print("   4通道文件:")
        for ch4_file in ch4_files[:2]:
            print(f"     {ch4_file.name}")
            if ch4_file.suffix == '.npz':
                try:
                    data = np.load(ch4_file)
                    for key in data.keys():
                        arr = data[key]
                        print(f"       {key}: {arr.shape}")
                except Exception as e:
                    print(f"       Error: {e}")
            elif ch4_file.suffix == '.png':
                try:
                    img = Image.open(ch4_file)
                    print(f"       PNG: {img.size}, mode: {img.mode}")
                except Exception as e:
                    print(f"       Error: {e}")

if __name__ == "__main__":
    analyze_mask_structure()