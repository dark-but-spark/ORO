import numpy as np
import os

def check_mask_values():
    """Check the actual values in mask files."""
    mask_dir = 'masks'
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])
    
    print("=" * 60)
    print("MASK VALUE ANALYSIS")
    print("=" * 60)
    
    # Check first 20 masks to get all unique values
    all_unique_values = set()
    value_counts = {}
    
    print("\nAnalyzing first 20 masks...")
    print("-" * 60)
    
    for i, mask_file in enumerate(mask_files[:20]):
        mask_path = os.path.join(mask_dir, mask_file)
        data = np.load(mask_path)
        mask = data['mask']
        
        # Get unique values
        unique_vals = np.unique(mask)
        all_unique_values.update(unique_vals.tolist())
        
        # Count occurrences
        for val in unique_vals:
            if val not in value_counts:
                value_counts[val] = 0
            value_counts[val] += 1
        
        if i < 5:  # Print details for first 5
            print(f"\nMask {i+1} ({mask_file}):")
            print(f"  Shape: {mask.shape}")
            print(f"  Unique values: {unique_vals}")
            print(f"  Value distribution:")
            for val in unique_vals[:10]:  # Show first 10 values
                count = np.sum(mask == val)
                percentage = (count / mask.size) * 100
                print(f"    {val}: {count:,} pixels ({percentage:.2f}%)")
    
    print("\n" + "=" * 60)
    print("SUMMARY (First 20 masks):")
    print("=" * 60)
    print(f"All unique values found: {sorted(all_unique_values)}")
    print(f"Number of unique values: {len(all_unique_values)}")
    print(f"\nValue frequency across masks:")
    for val in sorted(value_counts.keys()):
        print(f"  {val}: appears in {value_counts[val]}/20 masks")
    
    # Detailed check on one mask
    print("\n" + "=" * 60)
    print("DETAILED CHANNEL ANALYSIS (First mask):")
    print("=" * 60)
    sample_mask = np.load(os.path.join(mask_dir, mask_files[0]))['mask']
    
    for c in range(sample_mask.shape[2]):
        channel_data = sample_mask[:, :, c]
        unique_in_channel = np.unique(channel_data)
        print(f"\nChannel {c}:")
        print(f"  Unique values: {unique_in_channel}")
        print(f"  Min: {channel_data.min()}, Max: {channel_data.max()}")
        print(f"  Non-zero pixels: {np.sum(channel_data > 0):,} / {channel_data.size:,} ({np.sum(channel_data > 0) / channel_data.size * 100:.2f}%)")
        
        # Check if binary per channel
        is_binary = np.all(np.isin(unique_in_channel, [0, 1])) or \
                    np.all(np.isin(unique_in_channel, [0, 255]))
        print(f"  Is binary (0/1 or 0/255): {is_binary}")

if __name__ == "__main__":
    check_mask_values()
