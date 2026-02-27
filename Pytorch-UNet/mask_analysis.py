def analyze_and_fix_masks():
    """分析并修复mask数据处理"""
    print("=== Mask数据分析与修复建议 ===\n")
    
    # 分析数据分布
    from pathlib import Path
    mask_dir = Path('./data/masks/')
    
    npz_files = list(mask_dir.glob('*.npz'))
    png_files = list(mask_dir.glob('*.png'))
    
    single_channel = 0
    four_channel = 0
    
    print("文件类型统计:")
    for npz_file in npz_files:
        if '_4ch' in str(npz_file):
            four_channel += 1
        else:
            single_channel += 1
    
    for png_file in png_files:
        if '_4ch' in str(png_file):
            four_channel += 1
        else:
            single_channel += 1
    
    print(f"  单通道文件: {single_channel}")
    print(f"  4通道文件: {four_channel}")
    print(f"  总文件数: {len(npz_files) + len(png_files)}")
    
    # 检查具体的4通道数据
    print("\n4通道数据示例:")
    ch4_npz = list(mask_dir.glob('*_4ch.npz'))[:2]
    for file in ch4_npz:
        try:
            import numpy as np
            data = np.load(file)
            for key in data.keys():
                arr = data[key]
                print(f"  {file.name} - {key}: {arr.shape}")
                if arr.ndim == 3 and arr.shape[0] == 4:
                    print(f"    ✓ 正确的4通道格式")
                else:
                    print(f"    ✗ 非标准格式")
        except Exception as e:
            print(f"  {file.name} - 错误: {e}")
    
    print("\n=== 修复建议 ===")
    print("1. 对于4通道mask，应该保持 (4, H, W) 的形状")
    print("2. 对于单通道mask，应该squeeze到 (H, W) 的形状") 
    print("3. 当前的 torch.Size([640, 1, 640]) 是不正确的中间格式")
    print("4. 建议在训练时明确指定mask类型（单通道vs多通道）")

if __name__ == "__main__":
    analyze_and_fix_masks()