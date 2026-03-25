import numpy as np
import os

def check_masks():
    """Check mask files and their parameters."""
    mask_dir = 'masks'
    
    # Get all npz files
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])
    
    print("=" * 60)
    print("MASK FILE ANALYSIS")
    print("=" * 60)
    print(f"Total mask files: {len(mask_files)}")
    print()
    
    # Check first 10 masks
    print("DETAILED MASK INFO (First 10):")
    print("-" * 60)
    
    shapes = []
    channels = []
    dtypes = []
    
    for i, mask_file in enumerate(mask_files[:10]):
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            data = np.load(mask_path)
            mask = data['mask']
            
            shapes.append(mask.shape)
            if len(mask.shape) == 3:
                channels.append(mask.shape[2])
            else:
                channels.append(1)
            dtypes.append(mask.dtype)
            
            print(f"\n{i+1}. {mask_file}:")
            print(f"   Shape: {mask.shape}")
            print(f"   Dtype: {mask.dtype}")
            print(f"   Min: {mask.min()}, Max: {mask.max()}")
            print(f"   Unique values: {np.unique(mask)[:10]}{'...' if len(np.unique(mask)) > 10 else ''}")
            print(f"   Channels: {mask.shape[2] if len(mask.shape) == 3 else 1}")
            
            # Check if binary
            unique_vals = np.unique(mask)
            is_binary = np.all(np.isin(unique_vals, [0, 1, 0.0, 1.0]))
            print(f"   Is binary: {is_binary}")
            
        except Exception as e:
            print(f"\n{i+1}. {mask_file}: ERROR - {str(e)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Unique shapes: {set(shapes)}")
    print(f"Unique channels: {set(channels)}")
    print(f"Unique dtypes: {set(dtypes)}")
    print(f"Expected channels (from YAML): 4")
    print(f"Match expected: {4 in set(channels)}")

if __name__ == "__main__":
    check_masks()
