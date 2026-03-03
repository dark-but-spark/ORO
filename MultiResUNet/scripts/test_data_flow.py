import os
import cv2
import numpy as np
from tqdm import tqdm

# Import the actual data loading function
from dataloading import load_data

def test_data_loading():
    """
    Test the actual data loading process to find where values become zero.
    """
    print("=" * 60)
    print("DATA LOADING FLOW DIAGNOSIS")
    print("=" * 60)
    
    # Test loading with just 2 samples
    X, Y = load_data(limit=2)
    
    print("\n" + "=" * 60)
    print("LOADED DATA STATISTICS")
    print("=" * 60)
    
    print(f"\nX (Images) shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"X min: {X.min():.6f}")
    print(f"X max: {X.max():.6f}")
    print(f"X mean: {X.mean():.6f}")
    print(f"X is all zeros: {np.all(X == 0)}")
    
    print(f"\nY (Masks) shape: {Y.shape}")
    print(f"Y dtype: {Y.dtype}")
    print(f"Y min: {Y.min():.6f}")
    print(f"Y max: {Y.max():.6f}")
    print(f"Y mean: {Y.mean():.6f}")
    print(f"Y unique values: {np.unique(Y)}")
    
    # Check each image individually
    print("\n" + "=" * 60)
    print("INDIVIDUAL SAMPLE CHECK")
    print("=" * 60)
    
    for i in range(len(X)):
        print(f"\nSample {i+1}:")
        print(f"  Image - min: {X[i].min():.6f}, max: {X[i].max():.6f}, mean: {X[i].mean():.6f}")
        print(f"  Mask - min: {Y[i].min():.6f}, max: {Y[i].max():.6f}, mean: {Y[i].mean():.6f}")
        
        # Check if image is problematic
        if X[i].max() == 0:
            print(f"  ⚠ WARNING: Image {i+1} is all zeros!")
        elif X[i].max() < 0.1:
            print(f"  ⚠ WARNING: Image {i+1} has very low values (max={X[i].max():.6f})")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if np.all(X == 0):
        print("❌ CRITICAL ISSUE: All images are zeros after loading!")
        print("\nThis means the normalization or loading process is broken.")
        print("Expected: Images should have values in range [0, 1] after normalization")
        print("Actual: All values are 0.0")
    elif X.max() < 0.1:
        print("⚠ WARNING: Images have very low values")
        print("This suggests incorrect normalization")
    else:
        print("✓ Images appear to be loaded correctly")

if __name__ == "__main__":
    test_data_loading()