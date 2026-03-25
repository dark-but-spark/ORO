import numpy as np
import os
import yaml

def verify_dataset():
    """Verify dataset matches YAML configuration."""
    print("=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)
    
    # Load YAML config
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nYAML CONFIGURATION:")
    print("-" * 60)
    print(f"Train path: {config['train']}")
    print(f"Val path: {config['val']}")
    print(f"Test path: {config['test']}")
    print(f"Number of classes (nc): {config['nc']}")
    print(f"Class names: {config['names']}")
    
    # Check actual directories
    print("\n\nACTUAL DIRECTORY STRUCTURE:")
    print("-" * 60)
    
    dirs_to_check = ['imgs', 'masks', 'test', 'valid']
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            items = os.listdir(dir_name)
            print(f"✓ {dir_name}/ exists ({len(items)} items)")
            
            # Check subdirectories
            for item in items[:3]:
                item_path = os.path.join(dir_name, item)
                if os.path.isdir(item_path):
                    subitems = os.listdir(item_path)
                    print(f"   └─ {item}/ ({len(subitems)} items)")
        else:
            print(f"✗ {dir_name}/ MISSING")
    
    # Check mask files
    print("\n\nMASK FILES ANALYSIS:")
    print("-" * 60)
    mask_dir = 'masks'
    if os.path.exists(mask_dir):
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])
        print(f"Total .npz mask files: {len(mask_files)}")
        
        # Sample check
        if mask_files:
            sample_mask = np.load(os.path.join(mask_dir, mask_files[0]))['mask']
            print(f"\nSample mask properties:")
            print(f"  Shape: {sample_mask.shape}")
            print(f"  Channels: {sample_mask.shape[2] if len(sample_mask.shape) == 3 else 1}")
            print(f"  Dtype: {sample_mask.dtype}")
            print(f"  Value range: [{sample_mask.min()}, {sample_mask.max()}]")
            print(f"  Expected channels from YAML: {config['nc']}")
            print(f"  ✓ Channel count MATCHES" if sample_mask.shape[2] == config['nc'] else f"  ✗ Channel count MISMATCH")
    
    # Check image files
    print("\n\nIMAGE FILES ANALYSIS:")
    print("-" * 60)
    img_dir = 'imgs'
    if os.path.exists(img_dir):
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Total image files: {len(img_files)}")
        
        # Check if image count matches mask count
        if os.path.exists(mask_dir):
            print(f"Total mask files: {len(mask_files)}")
            if len(img_files) == len(mask_files):
                print(f"✓ Image and mask counts MATCH")
            else:
                print(f"✗ Image and mask counts MISMATCH: {len(img_files)} vs {len(mask_files)}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset()
