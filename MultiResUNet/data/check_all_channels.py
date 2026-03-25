import numpy as np
import os

def check_all_channels():
    """Check if all 4 channels have non-zero values across dataset."""
    mask_dir = 'masks'
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npz')])
    
    print("=" * 60)
    print("CHANNEL ACTIVITY ANALYSIS")
    print("=" * 60)
    
    # Check first 100 masks
    channel_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i, mask_file in enumerate(mask_files[:100]):
        mask_path = os.path.join(mask_dir, mask_file)
        data = np.load(mask_path)
        mask = data['mask']  # Shape: (640, 640, 4)
        
        # Check each channel
        for c in range(4):
            if np.sum(mask[:, :, c] > 0) > 0:
                channel_stats[c] += 1
    
    print(f"\nAnalysis of {len(mask_files[:100])} masks:")
    print("-" * 60)
    print("\nMasks with non-zero pixels per channel:")
    for c in range(4):
        count = channel_stats[c]
        percentage = (count / 100) * 100
        print(f"  Channel {c}: {count}/100 masks ({percentage:.1f}%)")
    
    # Check one mask with all channels active
    print("\n" + "=" * 60)
    print("SAMPLE MASK WITH ALL CHANNELS ACTIVE:")
    print("=" * 60)
    
    # Find a mask where all 4 channels have values
    for mask_file in mask_files[:50]:
        mask_path = os.path.join(mask_dir, mask_file)
        data = np.load(mask_path)
        mask = data['mask']
        
        all_channels_active = True
        print(f"\n{mask_file}:")
        for c in range(4):
            channel_data = mask[:, :, c]
            nonzero_count = np.sum(channel_data > 0)
            unique_vals = np.unique(channel_data[channel_data > 0])
            print(f"  Channel {c}: {nonzero_count:,} non-zero pixels, values: {unique_vals}")
            if nonzero_count == 0:
                all_channels_active = False
        
        if all_channels_active:
            break

if __name__ == "__main__":
    check_all_channels()
