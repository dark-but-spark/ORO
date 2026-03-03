import os
import cv2
import numpy as np
from tqdm import tqdm

def check_image_loading():
    """
    Check if images are being loaded correctly.
    """
    img_dir = 'data/imgs'
    img_files = sorted(next(os.walk(img_dir))[2])
    
    print("=" * 60)
    print("IMAGE LOADING DIAGNOSIS")
    print("=" * 60)
    
    # Check first 5 images
    for i, img_file in enumerate(img_files[:5]):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"✗ Image {i+1} ({img_file}): FAILED TO LOAD")
            continue
        
        print(f"\nImage {i+1} ({img_file}):")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Min: {img.min()}, Max: {img.max()}")
        print(f"  Mean: {img.mean():.2f}")
        
        # Check if image is all zeros
        if img.max() == 0:
            print(f"  ⚠ WARNING: Image is all zeros!")
        
        # Check file size
        file_size = os.path.getsize(img_path)
        print(f"  File size: {file_size:,} bytes")
        
        # Try to identify image format
        print(f"  Extension: {os.path.splitext(img_file)[1]}")
    
    print("\n" + "=" * 60)
    
    # Check if using correct path
    print(f"\nChecking image directory: {os.path.abspath(img_dir)}")
    print(f"Directory exists: {os.path.exists(img_dir)}")
    print(f"Total files in directory: {len(img_files)}")
    
    # Sample file paths
    print(f"\nSample file paths:")
    for i in range(min(3, len(img_files))):
        full_path = os.path.abspath(os.path.join(img_dir, img_files[i]))
        print(f"  {i+1}. {full_path}")
        print(f"     File exists: {os.path.exists(full_path)}")

if __name__ == "__main__":
    check_image_loading()