import os
import numpy as np
import cv2
from tqdm import tqdm

def diagnose_data():
    """
    Diagnose the training data to identify potential issues.
    """
    img_dir = 'data/imgs'
    mask_dir = 'data/masks'
    
    img_files = sorted(next(os.walk(img_dir))[2])
    mask_files = sorted(next(os.walk(mask_dir))[2])
    
    print("=" * 60)
    print("DATA DIAGNOSIS REPORT")
    print("=" * 60)
    print(f"Total images: {len(img_files)}")
    print(f"Total masks: {len(mask_files)}")
    print()
    
    # Check image dimensions and ranges
    print("IMAGE STATISTICS:")
    print("-" * 60)
    img_shapes = []
    img_mins = []
    img_maxs = []
    
    for i, img_file in enumerate(tqdm(img_files[:5], desc="Analyzing images")):
        img = cv2.imread(os.path.join(img_dir, img_file))
        img_shapes.append(img.shape)
        img_normalized = img / 255.0
        img_mins.append(img_normalized.min())
        img_maxs.append(img_normalized.max())
    
    print(f"Image shapes: {set(img_shapes)}")
    print(f"Image value range (normalized): [{min(img_mins):.4f}, {max(img_mins):.4f}]")
    print()
    
    # Check mask dimensions and ranges
    print("MASK STATISTICS:")
    print("-" * 60)
    mask_shapes = []
    mask_mins = []
    mask_maxs = []
    mask_channels = []
    
    for i, mask_file in enumerate(tqdm(mask_files[:5], desc="Analyzing masks")):
        mask = np.load(os.path.join(mask_dir, mask_file))['mask']
        mask_shapes.append(mask.shape)
        mask_normalized = mask / 255.0
        mask_mins.append(mask_normalized.min())
        mask_maxs.append(mask_normalized.max())
        if len(mask.shape) == 3:
            mask_channels.append(mask.shape[2])
        else:
            mask_channels.append(1)
    
    print(f"Mask shapes: {set(mask_shapes)}")
    print(f"Mask value range (normalized): [{min(mask_mins):.4f}, {max(mask_maxs):.4f}]")
    print(f"Mask channels: {set(mask_channels)}")
    print()
    
    # Check for binary masks
    print("BINARY MASK CHECK:")
    print("-" * 60)
    for i, mask_file in enumerate(mask_files[:3]):
        mask = np.load(os.path.join(mask_dir, mask_file))['mask']
        mask_normalized = mask / 255.0
        unique_vals = np.unique(mask_normalized)
        print(f"Mask {i+1} ({mask_file}):")
        print(f"  Shape: {mask.shape}")
        print(f"  Unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''} (total: {len(unique_vals)})")
        print(f"  Is binary (0/1 only): {np.all(np.isin(unique_vals, [0.0, 1.0]))}")
        print()
    
    # Check image-mask dimension matching
    print("IMAGE-MASK DIMENSION MATCHING:")
    print("-" * 60)
    for i in range(min(3, len(img_files))):
        img = cv2.imread(os.path.join(img_dir, img_files[i]))
        mask = np.load(os.path.join(mask_dir, mask_files[i]))['mask']
        
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2] if len(mask.shape) > 2 else mask.shape[:2]
        
        match = "✓" if (img_h == mask_h and img_w == mask_w) else "✗"
        print(f"Sample {i+1}: Image {img.shape} vs Mask {mask.shape} {match}")
    
    print()
    print("=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 60)
    
    if len(set(mask_shapes)) > 1:
        print("⚠ WARNING: Inconsistent mask shapes detected!")
        print("  → Consider resizing all masks to a uniform size")
    
    if max(mask_maxs) > 1.0 or min(mask_mins) < 0.0:
        print("⚠ WARNING: Mask values outside [0, 1] range after normalization!")
        print("  → Check normalization logic")
    
    if 1 in mask_channels and 4 in mask_channels:
        print("⚠ WARNING: Mixed single-channel and multi-channel masks!")
        print("  → Ensure consistent mask format across dataset")
    
    print("✓ All checks passed!" if not any([
        len(set(mask_shapes)) > 1,
        max(mask_maxs) > 1.0 or min(mask_mins) < 0.0,
        1 in mask_channels and 4 in mask_channels
    ]) else "")

if __name__ == "__main__":
    diagnose_data()
