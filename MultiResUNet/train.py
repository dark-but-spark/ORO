import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
import torch
import argparse
import gc
from datetime import datetime
import random

# Import the MultiResUNet model and utility functions
from pytorch.MultiResUNet import MultiResUnet, dice_coef, jacard, saveModel, evaluateModel, trainStep
from dataloading import load_data, split_data, create_datasets

# Define paths for data
IMAGE_DIR = 'data/imgs/'
MASK_DIR = 'data/masks/'


def check_memory_usage():
    """Check current system and GPU memory status"""
    import psutil
    
    print("=" * 60)
    print("Memory Status Check")
    print("=" * 60)
    
    # System memory
    mem = psutil.virtual_memory()
    print(f"System Memory:")
    print(f"  Total: {mem.total / 1024**3:.1f} GB")
    print(f"  Available: {mem.available / 1024**3:.1f} GB")
    print(f"  Used: {mem.used / 1024**3:.1f} GB ({mem.percent}%)")
    
    if mem.percent > 80:
        print(f"  ⚠ WARNING: High memory usage! Consider closing other applications")
    elif mem.percent > 90:
        print(f"  🚨 CRITICAL: Very high memory usage! OOM risk is high")
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"  Total: {gpu_mem:.1f} GB")
        print(f"  Allocated: {gpu_allocated:.2f} GB ({gpu_allocated/gpu_mem*100:.1f}%)")
        print(f"  Reserved: {gpu_reserved:.2f} GB ({gpu_reserved/gpu_mem*100:.1f}%)")
        
        if gpu_allocated/gpu_mem > 0.8:
            print(f"  ⚠ WARNING: High GPU memory usage!")
    
    print("")


def estimate_memory_requirements(data_limit, batch_size, image_size=(640, 640), channels=7):
    """Estimate memory requirements for training"""
    print("=" * 60)
    print("Memory Requirements Estimation")
    print("=" * 60)
    
    # Calculate per-sample memory
    bytes_per_sample = image_size[0] * image_size[1] * channels * 4  # float32 = 4 bytes
    mb_per_sample = bytes_per_sample / 1024**2
    
    print(f"Per Sample Memory:")
    print(f"  Image size: {image_size[0]}x{image_size[1]}")
    print(f"  Channels: {channels}")
    print(f"  Size per sample: {mb_per_sample:.2f} MB")
    
    # Total memory if loading all data
    total_mb = data_limit * mb_per_sample
    total_gb = total_mb / 1024
    
    print(f"\nFull Loading (NOT RECOMMENDED for large datasets):")
    print(f"  Samples: {data_limit}")
    print(f"  Total memory: {total_mb:.0f} MB ({total_gb:.1f} GB)")
    
    if total_gb > 8:
        print(f"  ⚠ WARNING: This will likely cause OOM!")
        print(f"  ✓ Recommendation: Use streaming data loading")
    
    # Streaming mode memory (only current batch + overhead)
    batch_mb = batch_size * mb_per_sample * 2  # input + mask
    overhead_mb = batch_mb * 0.3  # 30% overhead for gradients, optimizer states
    streaming_mb = batch_mb + overhead_mb + 500  # base overhead
    
    print(f"\nStreaming Mode (RECOMMENDED):")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory per batch: {batch_mb:.0f} MB")
    print(f"  Estimated total: ~{streaming_mb:.0f} MB ({streaming_mb/1024:.1f} GB)")
    print(f"  Memory savings: {(1 - streaming_mb/total_mb)*100:.1f}%")
    
    print("")


def diagnose_data_flow(args):
    """Run comprehensive data flow diagnosis"""
    print("\n" + "=" * 60)
    print("Data Flow Diagnosis")
    print("=" * 60)
    
    # Test data loading
    print("\n1. Testing data loading...")
    try:
        if args.data_limit and args.data_limit < 10:
            test_limit = args.data_limit
        else:
            test_limit = 5
        
        print(f"   Loading {test_limit} samples for testing...")
        X_test, Y_test = load_data(limit=test_limit, scale=args.scale, scale_factor=args.scale_factor)
        print(f"   ✓ Data loaded successfully")
        print(f"   - X shape: {X_test.shape}, dtype: {X_test.dtype}, range: [{X_test.min():.3f}, {X_test.max():.3f}]")
        print(f"   - Y shape: {Y_test.shape}, dtype: {Y_test.dtype}, range: [{Y_test.min():.3f}, {Y_test.max():.3f}]")
        
        
        # Check for all zeros
        if X_test.sum() == 0:
            print(f"   ⚠ WARNING: All input images are zero! Check data preprocessing")
        if Y_test.sum() == 0:
            print(f"   ⚠ WARNING: All masks are zero! Check mask generation")
        
        del X_test, Y_test
        gc.collect()
        
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        return False
    
    # Test dataset creation
    print("\n2. Testing streaming dataset creation...")
    try:
        train_ds, val_ds, n_train, n_val = create_datasets(
            limit=args.data_limit if args.data_limit else 10,
            scale=args.scale,
            scale_factor=args.scale_factor
        )
        print(f"   ✓ Dataset created successfully")
        print(f"   - Training samples: {n_train}")
        print(f"   - Validation samples: {n_val}")
        
        # Test data retrieval
        print("\n3. Testing batch retrieval...")
        from torch.utils.data import DataLoader
        test_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        
        for i, (img, mask) in enumerate(test_loader):
            print(f"   Batch {i+1}: img={img.shape}, mask={mask.shape}")
            if img.sum() == 0:
                print(f"   ⚠ WARNING: Batch contains all-zero images")
            if mask.sum() == 0:
                print(f"   ⚠ WARNING: Batch contains all-zero masks")
            if i >= 2:  # Test first 3 batches
                break
        
        del train_ds, val_ds, test_loader
        gc.collect()
        
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Data flow diagnosis completed successfully")
    return True


def parse_args():
    """
    Parse command line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train MultiResUNet for image segmentation")
    
    # Data loading arguments
    parser.add_argument('--data-limit', type=int, default=None, 
                        help='Number of samples to load for training (default: None). Use small values for quick testing.')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Proportion of data used for validation (default: 0.1)')
    
    # Image resizing arguments
    parser.add_argument('--scale', action='store_true',
                        help='Enable image scaling')
    parser.add_argument('--scale-factor', type=float, default=0.5,
                        help='Scale factor for images (default: 0.5). E.g., 0.5 reduces to 50%%, 1.5 increases to 150%%')
    
    # Model arguments
    parser.add_argument('--input-channels', type=int, default=3,
                        help='Number of input image channels (default: 3)')
    parser.add_argument('--output-channels', type=int, default=4,
                        help='Number of output segmentation channels (default: 4)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    
    # Optimization arguments
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0). Set to 0 to disable.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay (L2 regularization) for optimizer (default: 0)')
    
    # Data loading optimization
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of worker processes for data loading (default: 0). Increase to utilize more CPU cores')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Number of batches loaded in advance by each worker (default: 2)')
    
    # Repeat feeding for data augmentation
    parser.add_argument('--repeat-factor', type=int, default=1,
                        help='Number of times to repeat feed the same image with different augmentation (default: 1). '
                             'Setting this >1 enables dynamic augmentation where each repetition applies random transforms.')
    parser.add_argument('--train-augmentation', dest='train_augmentation', action='store_true',
                        help='Enable augmentation for training data in streaming dataset mode (default: enabled)')
    parser.add_argument('--no-train-augmentation', dest='train_augmentation', action='store_false',
                        help='Disable augmentation for training data in streaming dataset mode')
    parser.add_argument('--val-augmentation', action='store_true',
                        help='Enable augmentation for validation data (default: disabled, usually keep off)')
    parser.add_argument('--augmentation-strength', type=str, default='mild',
                        choices=['mild', 'moderate', 'strong'],
                        help='Augmentation profile for streaming dataset mode (default: mild)')
    parser.add_argument('--augmentation-curriculum', type=str, default='none',
                        choices=['none', 'linear', 'cosine'],
                        help='Schedule augmentation strength over epochs (default: none)')
    parser.add_argument('--curriculum-start-epoch', type=int, default=40,
                        help='Epoch where augmentation curriculum starts increasing strength (default: 40)')
    parser.add_argument('--curriculum-ramp-epochs', type=int, default=20,
                        help='Epochs used to ramp augmentation strength (default: 20)')
    parser.add_argument('--curriculum-max-aug-level', type=float, default=1.0,
                        help='Max interpolation level toward target augmentation profile (default: 1.0)')
    parser.add_argument('--curriculum-target-strength', type=str, default='moderate',
                        choices=['mild', 'moderate', 'strong'],
                        help='Target augmentation profile for curriculum (default: moderate)')
    parser.set_defaults(train_augmentation=True)
    
    # Logging and saving
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging during training')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoints during training')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save model checkpoints (default: models)')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--log-dir', type=str, default='runs/logs',
                        help='Directory for TensorBoard logs (default: runs/logs)')
    
    # Debugging options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    parser.add_argument('--check-data', action='store_true',
                        help='Run data validation checks before training')
    
    # Loss function options - for sparse/imbalanced datasets
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Use Focal Loss instead of BCE (recommended for sparse/imbalanced datasets)')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal Loss alpha parameter: weighting factor for positive/negative samples (default: 0.25)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss gamma parameter: focusing parameter to down-weight easy examples (default: 2.0)')
    parser.add_argument('--use-combined-loss', action='store_true',
                        help='Use combined BCE/Focal + Dice Loss for balanced training')
    parser.add_argument('--bce-weight', type=float, default=0.5,
                        help='Weight for BCE/Focal loss in combined loss (default: 0.5)')
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='Weight for Dice loss in combined loss (default: 0.5)')
    
    # Regularization parameters
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for regularization to prevent overfitting (default: 0.2)')
    
    # Learning rate scheduler options
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'exponential'],
                        help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler (default: 30)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor for StepLR/ExponentialLR schedulers (default: 0.1)')
    parser.add_argument('--lr-patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler (default: 10)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Stop after this many epochs without val Dice improvement. Set 0 to disable (default: 15)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Minimum val Dice improvement required to reset early stopping (default: 0.0)')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=0,
                        help='Disable early stopping before this epoch, useful with augmentation curriculum (default: 0)')
    
    # Device selection
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible split/shuffle/training (default: 42)')
    
    args = parser.parse_args()
    return args

# Main training function
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("=" * 60)
    print("MultiResUNet Training Configuration")
    print("=" * 60)
    print(f"Data Limit: {args.data_limit} samples")
    print(f"Validation Split: {args.validation_split:.1%}")
    print(f"Scale Enabled: {args.scale}")
    if args.scale:
        print(f"Scale Factor: {args.scale_factor} ({args.scale_factor*100:.0f}%)")
    print(f"Input Channels: {args.input_channels}")
    print(f"Output Channels: {args.output_channels}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gradient Clipping: {args.gradient_clip}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Device: {args.device}")
    print(f"Debug Mode: {args.debug}")
    print(f"Data Validation: {args.check_data}")
    print(f"TensorBoard: {args.tensorboard}")
    if args.tensorboard:
        print(f"Log Directory: {args.log_dir}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Prefetch Factor: {args.prefetch_factor}")
    print(f"Repeat Factor: {args.repeat_factor}")
    print(f"Train Augmentation: {args.train_augmentation}")
    print(f"Validation Augmentation: {args.val_augmentation}")
    print(f"Augmentation Strength: {args.augmentation_strength}")
    print(f"Augmentation Curriculum: {args.augmentation_curriculum}")
    if args.augmentation_curriculum != 'none':
        print(f"Curriculum: start=E{args.curriculum_start_epoch}, ramp={args.curriculum_ramp_epochs}, "
              f"max_level={args.curriculum_max_aug_level}, target={args.curriculum_target_strength}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    print(f"Early Stopping Min Delta: {args.early_stopping_min_delta}")
    print(f"Early Stopping Min Epochs: {args.early_stopping_min_epochs}")
    print(f"Random Seed: {args.seed}")
    print("=" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Memory safety check and recommendation
    if args.data_limit is not None:
        estimated_mb = args.data_limit * 640 * 640 * 7 * 4 / 1024 / 1024
        if args.data_limit > 500:
            print(f"\n🚨 LARGE DATASET DETECTED ({args.data_limit} samples)")
            print(f"   Estimated full loading memory: {estimated_mb:.0f} MB ({estimated_mb/1024:.1f} GB)")
            print(f"   ✓ FORCED: Using memory-efficient streaming loading")
            print(f"   ✓ Expected memory usage with streaming: <100 MB (99.7% savings)")
            print(f"   ⚠ WARNING: Full loading would cause OOM!\n")
        elif args.data_limit > 100:
            print(f"\nℹ INFO: Medium dataset ({args.data_limit} samples, ~{estimated_mb:.0f} MB)")
            print(f"   ✓ Recommendation: Use streaming mode for better memory efficiency\n")
    
    # Auto-enable scale for large datasets to reduce memory
    if args.data_limit and args.data_limit > 1000 and not args.scale:
        print(f"\n💡 AUTO-OPTIMIZATION: Large dataset detected")
        print(f"   Consider enabling scale to reduce memory usage:")
        print(f"   Recommended: --scale --scale-factor 0.5 (reduces to 320x320)")
        print(f"   This will save ~75% memory while maintaining good quality\n")
    
    # If repeat factor > 1, warn about increased computational cost
    if args.repeat_factor > 1:
        print(f"\n🔄 DATA AUGMENTATION: Enabled repeat feeding (factor: {args.repeat_factor})")
        print(f"   Each image will be fed {args.repeat_factor} times with different augmentations per epoch")
        print(f"   Note: This will increase training time by approximately {args.repeat_factor}x\n")
    
    # Check for GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Recommend batch size based on GPU memory
            if gpu_memory < 8 and args.batch_size > 4:
                print(f"⚠ WARNING: GPU has limited memory ({gpu_memory:.1f}GB). Consider reducing batch_size to 2-4")
            elif gpu_memory > 16 and args.batch_size < 8:
                print(f"ℹ INFO: GPU has plenty of memory ({gpu_memory:.1f}GB). Consider increasing batch_size to 16-32")
        except:
            pass
    
    print("")

    # Setup TensorBoard logging if enabled
    if args.tensorboard:
        # If log_dir is still the default 'runs/logs', create a unique subdirectory with timestamp
        if args.log_dir == 'runs/logs':
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a descriptive log directory name similar to run_training.sh
            log_dir = f"runs/tensorboard/train_{timestamp}_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}"
            if args.scale:
                log_dir += f"_scale{args.scale_factor}"
        else:
            log_dir = args.log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nTensorBoard logs will be saved to: {log_dir}")
    else:
        log_dir = None
    
    # Run memory check and diagnosis if in debug mode
    if args.debug or args.check_data:
        check_memory_usage()
        if args.data_limit:
            estimate_memory_requirements(args.data_limit, args.batch_size)
        diagnose_data_flow(args)
        print("\n" + "=" * 60)
        print("Starting Training After Diagnostics")
        print("=" * 60 + "\n")

    # Load data using memory-efficient approach
    print(f"\nLoading data...")
    
    # Option 1: Use new memory-efficient Dataset approach (recommended for large datasets)
    if args.data_limit is None or args.data_limit > 500:
        print("Using memory-efficient dataset loading (recommended for large datasets)...")
        from dataloading import create_datasets
        
        # Create datasets that will load data on-demand
        train_dataset, val_dataset, n_train, n_val = create_datasets(
            img_dir='data/imgs',
            mask_dir='data/masks',
            limit=args.data_limit,
            train_ratio=1.0 - args.validation_split,
            scale=args.scale,
            scale_factor=args.scale_factor,
            original_height=None,  # Will be determined automatically from first image
            original_width=None,   # Will be determined automatically from first image
            repeat_factor=args.repeat_factor,
            train_apply_augmentation=args.train_augmentation,
            val_apply_augmentation=args.val_augmentation,
            augmentation_strength=args.augmentation_strength,
            strong_aug_strength=args.curriculum_target_strength,
            augmentation_schedule_level=0.0,
            shuffle=True,
            seed=args.seed
        )
        
        # Create DataLoaders with OPTIMIZED parameters for memory efficiency
        from torch.utils.data import DataLoader
        
        # Calculate optimal num_workers based on CPU cores (leave 2 cores for system)
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(args.num_workers, max(1, cpu_count - 2))
        
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)

        def seed_worker(worker_id):
            worker_seed = args.seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=optimal_workers, 
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if optimal_workers > 0 else None,
            persistent_workers=False,  # CRITICAL: Disable to prevent memory leak
            drop_last=False,  # Keep last batch to avoid data waste
            worker_init_fn=seed_worker if optimal_workers > 0 else None,
            generator=loader_generator
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=optimal_workers, 
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if optimal_workers > 0 else None,
            persistent_workers=False,  # CRITICAL: Disable to prevent memory leak
            drop_last=False,
            worker_init_fn=seed_worker if optimal_workers > 0 else None,
            generator=loader_generator
        )
        
        print(f"✓ Training samples: {n_train}")
        print(f"✓ Validation samples: {n_val}")
        print(f"✓ Memory usage: Minimal (data loaded batch-by-batch)")
        print(f"✓ Optimized DataLoader config:")
        print(f"  - workers={optimal_workers} (auto-tuned from {args.num_workers})")
        print(f"  - prefetch={args.prefetch_factor}")
        print(f"  - persistent_workers=False (memory-safe)")
        print(f"  - pin_memory=True (GPU transfer optimization)")
        
        # Initialize model BEFORE training
        print(f"\nInitializing model...")
        model = MultiResUnet(
            input_channels=args.input_channels, 
            num_classes=args.output_channels
        ).to(device)
        
        print(f"Model architecture: MultiResUNet")
        print(f"  Input: {args.input_channels} channels")
        print(f"  Output: {args.output_channels} channels")
        
        # Train the model with DataLoaders
        print(f"\nStarting training...")
        print("-" * 60)
        
        history = trainStep(
            model, 
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            device=device,
            learning_rate=args.learning_rate,
            gradient_clip=args.gradient_clip,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            save_model=args.save_model,
            save_dir=args.save_dir,
            verbose=args.verbose,
            log_dir=log_dir,  # Pass TensorBoard log directory
            scale=args.scale,
            scale_factor=args.scale_factor,
            data_limit=args.data_limit,
            validation_split=args.validation_split,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            train_augmentation=args.train_augmentation,
            val_augmentation=args.val_augmentation,
            repeat_factor=args.repeat_factor,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_min_epochs=args.early_stopping_min_epochs,
            augmentation_curriculum=args.augmentation_curriculum,
            curriculum_start_epoch=args.curriculum_start_epoch,
            curriculum_ramp_epochs=args.curriculum_ramp_epochs,
            curriculum_max_aug_level=args.curriculum_max_aug_level,
            curriculum_base_strength=args.augmentation_strength,
            curriculum_target_strength=args.curriculum_target_strength,
            # Loss function configuration for sparse/imbalanced datasets
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            use_combined_loss=args.use_combined_loss,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            # Learning rate scheduler parameters
            lr_scheduler_type=args.lr_scheduler,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            lr_patience=args.lr_patience
        )
    
    else:
        # Option 2: Use traditional approach for small datasets (< 500 samples)
        print("Using traditional in-memory loading (suitable for small datasets)...")
        X, Y = load_data(limit=args.data_limit, 
                        scale=args.scale,
                        scale_factor=args.scale_factor)

        # Validate mask channels BEFORE any processing
        if args.debug or args.check_data:
            print(f"\nData Validation:")
            print(f"  Original Y shape: {Y.shape}")
            print(f"  Y sample unique values: {np.unique(Y[0])}")
            print(f"  Y value range: [{Y.min():.4f}, {Y.max():.4f}]")
            print(f"  Y positive pixel ratio: {Y.sum() / Y.size:.4f}")
        
        # Ensure Y has correct number of channels
        if Y.shape[-1] == 1:
            print("⚠ WARNING: Single channel mask detected. Duplicating to match output channels...")
            Y = np.concatenate([Y] * args.output_channels, axis=-1)
            print(f"After duplication Y shape: {Y.shape}")
        elif Y.shape[-1] != args.output_channels:
            raise ValueError(f"Expected {args.output_channels} channels in mask, got {Y.shape[-1]}")

        # Split data into training and validation sets
        print(f"\nSplitting data (validation={args.validation_split:.1%})...")
        X_train, X_val, Y_train, Y_val = split_data(X, Y, validation=args.validation_split)

        # Define the model
        print(f"\nInitializing model...")
        model = MultiResUnet(
            input_channels=args.input_channels, 
            num_classes=args.output_channels,
            dropout_rate=args.dropout_rate
        ).to(device)
        
        print(f"Model architecture: MultiResUNet")
        print(f"  Input: {args.input_channels} channels")
        print(f"  Output: {args.output_channels} channels")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Print data statistics
        if args.debug:
            print(f"\nData Statistics:")
            print(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
            print(f"  Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
            print(f"  Y_train positive ratio: {Y_train.sum() / Y_train.size:.4f}")

        # Train the model
        print(f"\nStarting training...")
        print("-" * 60)
        
        history = trainStep(
            model, 
            X_train, Y_train, 
            X_val, Y_val, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            device=device,
            learning_rate=args.learning_rate,
            gradient_clip=args.gradient_clip,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            save_model=args.save_model,
            save_dir=args.save_dir,
            verbose=args.verbose,
            log_dir=log_dir,  # Pass TensorBoard log directory
            scale=args.scale,
            scale_factor=args.scale_factor,
            data_limit=args.data_limit,
            validation_split=args.validation_split,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            train_augmentation=False,
            val_augmentation=False,
            repeat_factor=1,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_min_epochs=args.early_stopping_min_epochs,
            augmentation_curriculum=args.augmentation_curriculum,
            curriculum_start_epoch=args.curriculum_start_epoch,
            curriculum_ramp_epochs=args.curriculum_ramp_epochs,
            curriculum_max_aug_level=args.curriculum_max_aug_level,
            curriculum_base_strength=args.augmentation_strength,
            curriculum_target_strength=args.curriculum_target_strength,
            # Loss function configuration for sparse/imbalanced datasets
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            use_combined_loss=args.use_combined_loss,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            # Learning rate scheduler parameters
            lr_scheduler_type=args.lr_scheduler,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            lr_patience=args.lr_patience
        )
    
    print("-" * 60)
    print("Training complete!")
    
    # Save final model if requested
    if args.save_model:
        print(f"\nSaving final model to {args.save_dir}/")
        saveModel(model, args.save_dir)

if __name__ == "__main__":
    main()
