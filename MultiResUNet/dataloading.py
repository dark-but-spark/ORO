import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Optional


class SegmentationDataset(Dataset):
    """Custom Dataset for loading image segmentation data on-demand.
    
    This dataset loads images and masks from disk only when accessed,
    avoiding the need to load all data into memory at once.
    """
    
    def __init__(self, img_dir, mask_dir, img_files=None, mask_files=None,
                 limit=None, transform=None, scale=False, scale_factor=0.5,
                 original_height=None, original_width=None, repeat_factor=1,
                 apply_augmentation=True):
        """
        Args:
            img_dir (str): Directory containing image files
            mask_dir (str): Directory containing mask files
            img_files (list, optional): List of image filenames. If None, scans directory.
            mask_files (list, optional): List of mask filenames. If None, scans directory.
            limit (int, optional): Maximum number of samples to use
            transform (callable, optional): Optional transform to be applied on a sample
            scale (bool, optional): Whether to scale images and masks
            scale_factor (float, optional): Scale factor (e.g., 0.5 for 50% reduction)
            original_height (int, optional): Original height of images for augmentation
            original_width (int, optional): Original width of images for augmentation
            repeat_factor (int, optional): Number of times to repeat each image with different augmentation
            apply_augmentation (bool, optional): Whether to apply random augmentations to the data
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.scale = scale
        self.scale_factor = scale_factor
        self.original_height = original_height
        self.original_width = original_width
        self.repeat_factor = repeat_factor  # How many times to repeat each image with different augmentation
        self.apply_augmentation = apply_augmentation  # Whether to apply augmentations
        
        # Get file lists
        if img_files is None:
            img_files = next(os.walk(img_dir))[2]
            img_files.sort()
        
        if mask_files is None:
            mask_files = next(os.walk(mask_dir))[2]
            mask_files.sort()
        
        # Apply limit
        if limit:
            img_files = img_files[:limit]
            mask_files = mask_files[:limit]
        
        # Expand file lists if repeat factor > 1
        if repeat_factor > 1:
            self.img_files = []
            self.mask_files = []
            for _ in range(repeat_factor):
                self.img_files.extend(img_files)
                self.mask_files.extend(mask_files)
        else:
            self.img_files = img_files
            self.mask_files = mask_files
        
        print(f"Dataset created with {len(self.img_files)} samples")
        if scale:
            print(f"  Scale enabled: {scale_factor*100:.0f}%")
        if original_height and original_width:
            print(f"  Original dimensions: {original_width}x{original_height}")
        if repeat_factor > 1:
            print(f"  Repeat factor: {repeat_factor} (effective samples: {len(img_files) * repeat_factor})")
        print(f"  Augmentation: {'Enabled' if apply_augmentation else 'Disabled'}")
    
    def __len__(self):
        return len(self.img_files)
    
    def apply_random_augmentations(self, img, mask):
        """Apply random augmentations to both image and mask consistently"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        
        # Random rotation (any angle) - with valid region extraction
        if random.random() > 0.7:  # 30% chance to rotate
            # Generate a random angle in degrees
            angle = random.uniform(-45, 45)  # Reduced range to avoid extreme distortions
            
            # Get image dimensions
            h, w = img.shape[0], img.shape[1]
            
            # Compute rotation matrix
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to image using linear interpolation for better quality
            img_rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_REPLICATE)
            
            # Apply the same rotation to mask
            # For masks, use nearest neighbor interpolation to preserve discrete values
            mask_rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, 
                                        borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate the valid region after rotation to avoid boundary artifacts
            # For a rotation of angle θ, the valid central region has sides scaled by cos(θ) + sin(θ)
            angle_rad = abs(angle) * np.pi / 180.0  # Convert to radians
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Calculate the size of the valid central rectangle after rotation
            new_w = int(w * cos_a - h * sin_a)
            new_h = int(h * cos_a - w * sin_a)
            
            # Ensure positive dimensions
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            # Calculate the coordinates of the valid region
            x_start = (w - new_w) // 2
            y_start = (h - new_h) // 2
            
            # Extract the valid region from rotated images
            img = img_rotated[y_start:y_start+new_h, x_start:x_start+new_w]
            mask = mask_rotated[y_start:y_start+new_h, x_start:x_start+new_w]
            
            # Resize back to original dimensions to maintain consistent output size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            if len(mask.shape) == 3:
                # Multi-channel mask
                resized_mask = np.zeros((h, w, mask.shape[2]), dtype=mask.dtype)
                for c in range(mask.shape[2]):
                    resized_mask[:, :, c] = cv2.resize(mask[:, :, c], (w, h), 
                                                      interpolation=cv2.INTER_NEAREST)
                mask = resized_mask
            else:
                # Single channel mask
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            # Adjust brightness by factor between 0.8 and 1.2
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            # Adjust contrast by factor between 0.8 and 1.2
            factor = random.uniform(0.8, 1.2)
            img_mean = img.mean(axis=(0, 1), keepdims=True)
            img = np.clip((img - img_mean) * factor + img_mean, 0, 1)
        
        # Random Gaussian noise
        if random.random() > 0.7:  # 30% chance to add noise
            noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        
        # Random zoom - with improved handling
        if random.random() > 0.7:  # 30% chance to zoom
            zoom_factor = random.uniform(0.9, 1.1)  # More conservative zoom to reduce border effects
                                               # Changed from (0.8, 1.2) to (0.9, 1.1)
            
            # Calculate new size
            h, w = img.shape[0], img.shape[1]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            # Resize image
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Place the resized image/mask in the center of the original canvas
            if zoom_factor <= 1.0:
                # Zoom in - place resized image in the center
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                
                img_new = np.zeros_like(img)
                mask_new = np.zeros_like(mask)
                
                img_new[top:top+new_h, left:left+new_w] = img_resized
                mask_new[top:top+new_h, left:left+new_w] = mask_resized
                
                img, mask = img_new, mask_new
            else:
                # Zoom out - crop the center portion to avoid introducing new boundary pixels
                top = (new_h - h) // 2
                left = (new_w - w) // 2
                
                img = img_resized[top:top+h, left:left+w]
                mask = mask_resized[top:top+h, left:left+w]
        
        return img, mask
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        
        # Store original dimensions if not provided
        h, w = img.shape[0], img.shape[1]
        if self.original_height is None or self.original_width is None:
            orig_h, orig_w = h, w
        else:
            orig_h, orig_w = self.original_height, self.original_width
            
        img = img / 255.0  # Normalize to [0, 1]
        img = img.astype(np.float32)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask_data = np.load(mask_path)
        mask = mask_data['mask']
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = mask.astype(np.float32)
        
        # Scale if enabled - BEFORE augmentations
        if self.scale:
            # Calculate new dimensions based on scale factor
            h, w = img.shape[0], img.shape[1]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            
            # Validate dimensions are reasonable
            if new_w <= 0 or new_h <= 0:
                raise ValueError(f"Scale factor {self.scale_factor} results in invalid dimensions: {new_w}x{new_h}")
            
            # Use cubic interpolation for images (better quality for continuous values)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Use nearest neighbor interpolation for masks (preserve discrete values)
            if len(mask.shape) == 3:
                # Multi-channel mask: resize each channel separately with nearest neighbor
                resized_mask = np.zeros((new_h, new_w, mask.shape[2]), dtype=np.float32)
                for c in range(mask.shape[2]):
                    resized_mask[:, :, c] = cv2.resize(mask[:, :, c], (new_w, new_h), 
                                                       interpolation=cv2.INTER_NEAREST)
                mask = resized_mask
            else:
                # Single channel mask
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply random augmentations if enabled - AFTER scaling
        if self.apply_augmentation:
            img, mask = self.apply_random_augmentations(img, mask)
        
        # Convert to CHW format for PyTorch
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        mask = mask.transpose(2, 0, 1) if len(mask.shape) == 3 else mask[np.newaxis, :, :]
        
        return torch.from_numpy(img), torch.from_numpy(mask)


def load_data(limit=None, scale=False, scale_factor=0.5):
    img_files = next(os.walk('data/imgs'))[2]
    label_files = next(os.walk('data/masks'))[2]

    img_files.sort()
    label_files.sort()

    if limit:
        img_files = img_files[:limit]
        label_files = label_files[:limit]

    print(f"Number of image files: {len(img_files)}")
    print(f"Number of label files: {len(label_files)}")
    if scale:
        print(f"Scaling images by factor {scale_factor*100:.0f}%")
    
    X = []
    Y = []

    for i in tqdm(img_files):
        # Load and preprocess image
        img = cv2.imread(os.path.join('data/imgs', i))
        
        # Resize if enabled
        if scale:
            # Calculate new dimensions based on scale factor
            h, w = img.shape[0], img.shape[1]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            # Validate dimensions
            if new_w <= 0 or new_h <= 0:
                raise ValueError(f"Scale factor {scale_factor} results in invalid dimensions: {new_w}x{new_h}")
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        img = img / 255.0  # Normalize image
        X.append(img)
    print(f"Finished loading images. {len(X)} images loaded.")
    for i in tqdm(label_files):
        # Load and preprocess mask from .npz file
        mask = np.load(os.path.join('data/masks', i))['mask']
        
        # Resize mask if enabled
        if scale:
            # Calculate new dimensions based on scale factor
            h, w = mask.shape[0], mask.shape[1]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            # Validate dimensions
            if new_w <= 0 or new_h <= 0:
                raise ValueError(f"Scale factor {scale_factor} results in invalid dimensions: {new_w}x{new_h}")
            
            # Handle multi-channel masks properly
            if len(mask.shape) == 3:
                # Multi-channel mask: resize each channel separately
                resized_mask = np.zeros((new_h, new_w, mask.shape[2]), dtype=np.float32)
                for c in range(mask.shape[2]):
                    resized_mask[:, :, c] = cv2.resize(mask[:, :, c], (new_w, new_h), 
                                                       interpolation=cv2.INTER_NEAREST)
                mask = resized_mask
            else:
                # Single channel mask
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        mask = mask / 255.0  # Normalize mask
        Y.append(mask)
    print(f"Finished loading masks. {len(Y)} masks loaded.")
    X = np.array(X, dtype='float32')  # Ensure data type is float32
    Y = np.array(Y, dtype='float32')
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")    
    return X, Y


def create_datasets(img_dir='data/imgs', mask_dir='data/masks',
                   train_ratio=0.9, limit=None, val_ratio=0.1,
                   scale=False, scale_factor=0.5, original_height=None, original_width=None,
                   repeat_factor=1, train_apply_augmentation=True, val_apply_augmentation=False,
                   shuffle=True, seed=42):
    """Create training and validation datasets without loading all data into memory.
    
    This function creates dataset objects that will load data on-demand,
    significantly reducing memory usage compared to load_data().
    
    Args:
        img_dir (str): Directory containing image files
        mask_dir (str): Directory containing mask files
        train_ratio (float): Ratio of data to use for training
        limit (int, optional): Maximum total samples to use
        val_ratio (float): Ratio of training data to use for validation
        scale (bool, optional): Whether to scale images and masks
        scale_factor (float, optional): Scale factor (e.g., 0.5 for 50% reduction)
        original_height (int, optional): Original height of images for augmentation
        original_width (int, optional): Original width of images for augmentation
        repeat_factor (int, optional): Number of times to repeat each image with different augmentation
        train_apply_augmentation (bool): Whether to apply augmentations to training data
        val_apply_augmentation (bool): Whether to apply augmentations to validation data
        shuffle (bool): Whether to shuffle file pairs before splitting train/validation
        seed (int): Random seed used for train/validation split shuffling
    
    Returns:
        tuple: (train_dataset, val_dataset, n_train, n_val)
    """
    import torch
    
    # Get all file lists
    img_files = next(os.walk(img_dir))[2]
    mask_files = next(os.walk(mask_dir))[2]
    
    img_files.sort()
    mask_files.sort()
    
    # Apply limit
    if limit:
        img_files = img_files[:limit]
        mask_files = mask_files[:limit]
    
    paired_files = list(zip(img_files, mask_files))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paired_files)

    if paired_files:
        img_files, mask_files = zip(*paired_files)
        img_files = list(img_files)
        mask_files = list(mask_files)
    else:
        img_files, mask_files = [], []

    n_total = len(img_files)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    
    # Split file lists
    train_img_files = img_files[:n_train]
    train_mask_files = mask_files[:n_train]
    val_img_files = img_files[n_train:]
    val_mask_files = mask_files[n_train:]
    
    print(f"Total samples: {n_total}")
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    if scale:
        print(f"  Scale factor: {scale_factor*100:.0f}%")
    
    # Create datasets (these won't load data until accessed)
    train_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_files=train_img_files,
        mask_files=train_mask_files,
        limit=limit,
        transform=None,
        scale=scale,
        scale_factor=scale_factor,
        original_height=original_height,
        original_width=original_width,
        repeat_factor=repeat_factor,
        apply_augmentation=train_apply_augmentation
    )
    
    val_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_files=val_img_files,
        mask_files=val_mask_files,
        limit=limit,
        transform=None,
        scale=scale,
        scale_factor=scale_factor,
        original_height=original_height,
        original_width=original_width,
        repeat_factor=1,  # Don't repeat validation images, only for training
        apply_augmentation=val_apply_augmentation  # Usually we don't want augmentation on validation data
    )
    
    return train_dataset, val_dataset, n_train, n_val


def split_data(X, Y, validation=0.1, random_state=42):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation, random_state=random_state)
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    return X_train, X_val, Y_train, Y_val


if __name__ == "__main__": 
    # Test the new memory-efficient approach
    print("Testing memory-efficient dataset loading...")
    print("=" * 60)
    
    # Method 1: Old approach (loads all data)
    print("\nMethod 1: Loading all data into memory (OLD APPROACH)")
    print("-" * 60)
    X, Y = load_data(limit=100)
    print(f"Memory used: X={X.nbytes/1024/1024:.1f}MB, Y={Y.nbytes/1024/1024:.1f}MB")
    
    # Method 2: New approach (lazy loading)
    print("\n\nMethod 2: On-demand loading (NEW APPROACH)")
    print("-" * 60)
    train_ds, val_ds, n_train, n_val = create_datasets(limit=100)
    print(f"✓ Datasets created without loading data into memory!")
    print(f"✓ Data will be loaded batch-by-batch during training")
    print(f"✓ Memory savings: ~{100*640*640*3*4/1024/1024:.0f}MB for 100 samples")
