import os
import json
import numpy as np
import torch
import argparse
import gc
import random
import sys
import atexit
from pathlib import Path
from datetime import datetime

# Import the MultiResUNet model and utility functions
from pytorch.MultiResUNet import (
    MultiResUnet,
    dice_coef,
    jacard,
    per_class_segmentation_metrics,
    predict_prob_with_tta,
    saveModel,
    evaluateModel,
    trainStep,
)
from dataloading import load_data, split_data, create_datasets, create_fixed_datasets, create_single_dataset

# Define paths for data
IMAGE_DIR = 'data/train/images'
MASK_DIR = 'data/train/masks'


class TeeStream:
    """Mirror stdout/stderr to a run log while preserving normal console output."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, 'encoding', 'utf-8')

    def write(self, data):
        self.stream.write(data)
        if not self.log_file.closed:
            self.log_file.write(data)

    def flush(self):
        self.stream.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def close(self):
        self.flush()


def _experiment_name_from_args(args):
    for candidate in (args.save_dir, args.log_dir):
        if candidate:
            name = Path(candidate).name
            if name and name not in ('models', 'logs', 'tensorboard'):
                return name
    return f"train_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}"


def _pid_is_running(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_experiment_lock(experiment_name):
    lock_dir = Path('runs') / '.active'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{experiment_name}.lock"
    pid = os.getpid()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as lock_file:
                lock_file.write(f"{pid}\n")
            break
        except FileExistsError:
            try:
                existing_pid = int(lock_path.read_text(encoding='utf-8').strip().splitlines()[0])
            except (OSError, ValueError, IndexError):
                existing_pid = -1

            if _pid_is_running(existing_pid):
                raise RuntimeError(
                    f"Experiment '{experiment_name}' is already running with PID {existing_pid}. "
                    "Refusing to create a second timestamped run directory."
                )

            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def release_lock():
        try:
            if lock_path.read_text(encoding='utf-8').strip().splitlines()[0] == str(pid):
                lock_path.unlink()
        except (OSError, IndexError):
            pass

    atexit.register(release_lock)
    return lock_path


def setup_run_outputs(args):
    """
    Keep all artifacts from one invocation under a single timestamped run directory.

    This intentionally overrides the old split layout where --save-dir and --log-dir
    could point at unrelated folders. Existing command lines still work because their
    experiment name is reused to name the run directory.
    """
    experiment_name = _experiment_name_from_args(args)
    lock_path = _acquire_experiment_lock(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path('runs') / f"{experiment_name}_{timestamp}"
    model_dir = run_dir / 'models'
    tensorboard_dir = run_dir / 'tensorboard'
    log_dir = run_dir / 'logs'
    history_dir = run_dir / 'history'

    for directory in (model_dir, log_dir, history_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.tensorboard:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

    original_save_dir = args.save_dir
    original_log_dir = args.log_dir
    args.run_dir = str(run_dir)
    args.save_dir = str(model_dir)
    args.metadata_dir = str(history_dir)
    if args.tensorboard:
        args.log_dir = str(tensorboard_dir)

    stdout_log = open(log_dir / 'training.log', 'a', encoding='utf-8', buffering=1)
    stderr_log = open(log_dir / 'training.err', 'a', encoding='utf-8', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, stdout_log)
    sys.stderr = TeeStream(original_stderr, stderr_log)

    def close_run_logs():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        for log_file in (stdout_log, stderr_log):
            try:
                if not log_file.closed:
                    log_file.flush()
                    log_file.close()
            except OSError:
                pass

    atexit.register(close_run_logs)

    print("=" * 60)


def _make_loader(dataset, args, shuffle=False, batch_size=None):
    from torch.utils.data import DataLoader

    cpu_count = os.cpu_count() or 4
    optimal_workers = min(args.num_workers, max(1, cpu_count - 2))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    def seed_worker(worker_id):
        # Varies deterministically when workers are recreated each epoch.
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else args.batch_size,
        shuffle=shuffle,
        num_workers=optimal_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if optimal_workers > 0 else None,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=seed_worker if optimal_workers > 0 else None,
        generator=loader_generator,
    ), optimal_workers


def _save_split_manifest(args, train_dataset=None, val_dataset=None, test_dataset=None):
    manifest = {
        "split_mode": args.split_mode,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "train_patch": {
            "size": args.train_patch_size,
            "positive_probability": args.patch_positive_probability,
            "class_indices": args.patch_class_indices,
            "min_positive_pixels": args.patch_min_positive_pixels,
            "center_jitter": args.patch_center_jitter,
        },
        "paths": {
            "train_img_dir": args.train_img_dir,
            "train_mask_dir": args.train_mask_dir,
            "val_img_dir": args.val_img_dir,
            "val_mask_dir": args.val_mask_dir,
            "test_img_dir": args.test_img_dir,
            "test_mask_dir": args.test_mask_dir,
        },
        "splits": {},
    }

    for split_name, dataset in (
        ("train", train_dataset),
        ("valid", val_dataset),
        ("test", test_dataset),
    ):
        if dataset is None:
            continue
        manifest["splits"][split_name] = {
            "count": len(dataset),
            "img_dir": dataset.img_dir,
            "mask_dir": dataset.mask_dir,
            "image_files": list(dataset.img_files),
            "mask_files": list(dataset.mask_files),
        }

    out_dir = Path(args.metadata_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Split manifest saved to: {manifest_path}")
    return manifest_path


def evaluate_loader(model, loader, device, tta_mode='none', threshold=0.5, metric_ignore_classes=None):
    model.eval()
    total_dice = 0.0
    total_jaccard = 0.0
    total_ignore_dice = 0.0
    total_ignore_jaccard = 0.0
    num_batches = 0
    num_samples = 0
    class_intersection_sum = None
    class_true_sum = None
    class_pred_sum = None

    with torch.no_grad():
        for images, targets in loader:
            batch_samples = targets.size(0)
            images = images.to(device)
            targets = targets.to(device)
            probs = predict_prob_with_tta(model, images, tta_mode)
            preds = (probs >= threshold).float()

            total_dice += dice_coef(targets, preds).item() * batch_samples
            total_jaccard += jacard(targets, preds).item() * batch_samples
            targets_by_class = targets.reshape(targets.size(0), targets.size(1), -1)
            preds_by_class = preds.reshape(preds.size(0), preds.size(1), -1)
            class_intersection = (targets_by_class * preds_by_class).sum(dim=(0, 2)).detach().cpu()
            class_true = targets_by_class.sum(dim=(0, 2)).detach().cpu()
            class_pred = preds_by_class.sum(dim=(0, 2)).detach().cpu()
            class_intersection_sum = class_intersection if class_intersection_sum is None else class_intersection_sum + class_intersection
            class_true_sum = class_true if class_true_sum is None else class_true_sum + class_true
            class_pred_sum = class_pred if class_pred_sum is None else class_pred_sum + class_pred

            if metric_ignore_classes:
                keep = [
                    idx for idx in range(targets.size(1))
                    if idx not in set(metric_ignore_classes)
                ]
                total_ignore_dice += dice_coef(targets[:, keep], preds[:, keep]).item() * batch_samples
                total_ignore_jaccard += jacard(targets[:, keep], preds[:, keep]).item() * batch_samples

            num_batches += 1
            num_samples += batch_samples

    if num_samples == 0:
        raise ValueError("Evaluation loader is empty")

    smooth = 1e-6
    class_dice = (2.0 * class_intersection_sum + smooth) / (class_true_sum + class_pred_sum + smooth)
    class_jaccard = (class_intersection_sum + smooth) / (class_true_sum + class_pred_sum - class_intersection_sum + smooth)
    metrics = {
        "dice": total_dice / num_samples,
        "jaccard": total_jaccard / num_samples,
        "threshold": threshold,
        "tta": tta_mode,
        "batches": num_batches,
        "samples": num_samples,
        "class_dice": class_dice.tolist(),
        "class_jaccard": class_jaccard.tolist(),
    }
    if metric_ignore_classes:
        metrics["metric_ignore_classes"] = list(metric_ignore_classes)
        metrics["dice_ignore_classes"] = total_ignore_dice / num_samples
        metrics["jaccard_ignore_classes"] = total_ignore_jaccard / num_samples
    return metrics


def run_final_test_evaluation(args, model, device):
    if not args.run_test_after_training:
        return None
    if not os.path.isdir(args.test_img_dir) or not os.path.isdir(args.test_mask_dir):
        print(f"WARNING: test directories not found, skipping test evaluation: {args.test_img_dir}, {args.test_mask_dir}")
        return None

    print("\nRunning final test evaluation on fixed test split...")
    test_dataset = create_single_dataset(
        img_dir=args.test_img_dir,
        mask_dir=args.test_mask_dir,
        limit=args.test_limit,
        scale=args.scale,
        scale_factor=args.scale_factor,
        apply_augmentation=False,
        augmentation_strength=args.augmentation_strength,
    )
    test_loader, _ = _make_loader(
        test_dataset,
        args,
        shuffle=False,
        batch_size=args.eval_batch_size,
    )

    best_model_path = Path(args.save_dir) / "best_model.pth"
    if args.save_model and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        print(f"Loaded best checkpoint for test evaluation: {best_model_path}")
    else:
        print("Best checkpoint not found or --save-model disabled; evaluating current in-memory model.")

    metrics = evaluate_loader(
        model,
        test_loader,
        device,
        tta_mode=args.test_tta,
        threshold=args.test_threshold,
        metric_ignore_classes=args.metric_ignore_classes,
    )
    metrics["split"] = "test"
    metrics["samples"] = len(test_dataset)
    metrics["img_dir"] = args.test_img_dir
    metrics["mask_dir"] = args.test_mask_dir

    out_dir = Path(args.metadata_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Test Dice: {metrics['dice']:.4f}, Test Jaccard: {metrics['jaccard']:.4f}")
    if "dice_ignore_classes" in metrics:
        print(f"Test Dice ignore {metrics['metric_ignore_classes']}: {metrics['dice_ignore_classes']:.4f}")
    print(f"Test metrics saved to: {metrics_path}")
    return metrics
    print("Run output layout")
    print("=" * 60)
    print(f"Process ID: {os.getpid()}")
    print(f"Run Lock: {lock_path}")
    print(f"Run Directory: {args.run_dir}")
    print(f"Original Save Directory: {original_save_dir}")
    print(f"Model Directory: {args.save_dir}")
    print(f"Original TensorBoard Directory: {original_log_dir}")
    print(f"TensorBoard Directory: {args.log_dir if args.tensorboard else 'disabled'}")
    print(f"Python Log File: {log_dir / 'training.log'}")
    print(f"Python Error Log File: {log_dir / 'training.err'}")
    print(f"History Directory: {args.metadata_dir}")
    print("=" * 60)


def create_model(args):
    """Create the requested segmentation model.

    The default keeps the existing MultiResUNet path unchanged. The SMP path is
    optional so existing training does not require extra dependencies.
    """
    if args.model_architecture == 'multiresunet':
        model = MultiResUnet(
            input_channels=args.input_channels,
            num_classes=args.output_channels,
            dropout_rate=args.dropout_rate
        )
        model_name = 'MultiResUNet'
    elif args.model_architecture == 'smp_unet':
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required for --model-architecture smp_unet. "
                "Install it on the server with: pip install segmentation-models-pytorch"
            )
        encoder_weights = args.encoder_weights
        if encoder_weights is not None and encoder_weights.lower() in ('none', 'null', 'false'):
            encoder_weights = None
        model = smp.Unet(
            encoder_name=args.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=args.input_channels,
            classes=args.output_channels,
            activation=None
        )
        model_name = f"SMP-Unet({args.encoder_name}, weights={encoder_weights})"
    else:
        raise ValueError(f"Unsupported model architecture: {args.model_architecture}")

    return model, model_name


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
                        help='Proportion of data used for validation in random split mode (default: 0.1)')
    parser.add_argument('--split-mode', type=str, default='fixed',
                        choices=['fixed', 'random'],
                        help='Data split mode. fixed uses explicit train/valid/test dirs; random uses old img/mask split (default: fixed)')
    parser.add_argument('--train-img-dir', type=str, default='data/train/images',
                        help='Fixed split training image directory')
    parser.add_argument('--train-mask-dir', type=str, default='data/train/masks',
                        help='Fixed split training .npz mask directory')
    parser.add_argument('--val-img-dir', type=str, default='data/valid/images',
                        help='Fixed split validation image directory')
    parser.add_argument('--val-mask-dir', type=str, default='data/valid/masks',
                        help='Fixed split validation .npz mask directory')
    parser.add_argument('--test-img-dir', type=str, default='data/test/images',
                        help='Fixed split test image directory')
    parser.add_argument('--test-mask-dir', type=str, default='data/test/masks',
                        help='Fixed split test .npz mask directory')
    parser.add_argument('--run-test-after-training', dest='run_test_after_training', action='store_true',
                        help='Evaluate best checkpoint on fixed test split after training (default: enabled)')
    parser.add_argument('--no-test-after-training', dest='run_test_after_training', action='store_false',
                        help='Disable final fixed test evaluation')
    parser.add_argument('--test-limit', type=int, default=None,
                        help='Optional sample limit for final test evaluation')
    parser.add_argument('--test-threshold', type=float, default=0.5,
                        help='Prediction threshold for final test evaluation (default: 0.5)')
    parser.add_argument('--test-tta', type=str, default='none',
                        choices=['none', 'flips'],
                        help='TTA mode for final test evaluation (default: none)')
    parser.set_defaults(run_test_after_training=True)
    
    # Image resizing arguments
    parser.add_argument('--scale', action='store_true',
                        help='Enable image scaling')
    parser.add_argument('--scale-factor', type=float, default=0.5,
                        help='Scale factor for images (default: 0.5). E.g., 0.5 reduces to 50%%, 1.5 increases to 150%%')
    parser.add_argument('--train-patch-size', type=int, default=0,
                        help='Native-resolution square patches for training only; 0 keeps whole images (default: 0)')
    parser.add_argument('--patch-positive-probability', type=float, default=0.75,
                        help='Probability of centering a training patch near a positive mask pixel (default: 0.75)')
    parser.add_argument('--patch-class-indices', type=int, nargs='+', default=None,
                        help='Mask channels eligible for positive-centered patches; default uses all channels')
    parser.add_argument('--patch-min-positive-pixels', type=int, default=1,
                        help='Minimum positive pixels for a channel to drive ROI patch sampling (default: 1)')
    parser.add_argument('--patch-center-jitter', type=float, default=0.25,
                        help='Patch-center jitter as a fraction of patch size, in [0, 0.5] (default: 0.25)')
    
    # Model arguments
    parser.add_argument('--input-channels', type=int, default=3,
                        help='Number of input image channels (default: 3)')
    parser.add_argument('--output-channels', type=int, default=4,
                        help='Number of output segmentation channels (default: 4)')
    parser.add_argument('--model-architecture', type=str, default='multiresunet',
                        choices=['multiresunet', 'smp_unet'],
                        help='Model architecture: existing multiresunet or optional segmentation_models_pytorch Unet')
    parser.add_argument('--encoder-name', type=str, default='resnet34',
                        help='SMP encoder name when --model-architecture smp_unet (default: resnet34)')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='SMP encoder weights, e.g. imagenet or None. Only used by smp_unet (default: imagenet)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training (default: 2)')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='Validation and final-test batch size; defaults to --batch-size')
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
                        choices=['none', 'linear', 'cosine', 'adaptive'],
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
    parser.add_argument('--curriculum-level-step', type=float, default=0.05,
                        help='Adaptive curriculum level increment when validation recovers (default: 0.05)')
    parser.add_argument('--curriculum-adapt-window', type=int, default=3,
                        help='Recent validation Dice window for adaptive curriculum decisions (default: 3)')
    parser.add_argument('--curriculum-adapt-tolerance', type=float, default=0.002,
                        help='Allowed Dice drop from stage reference before holding adaptive level (default: 0.002)')
    parser.add_argument('--curriculum-min-level-epochs', type=int, default=4,
                        help='Minimum epochs to train at each adaptive curriculum level before increasing (default: 4)')
    parser.set_defaults(train_augmentation=True)
    
    # Logging and saving
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging during training')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model checkpoints during training')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save model checkpoints (default: models)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save epoch checkpoints every N epochs. Set 0 to disable periodic checkpoints (default: 10)')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--log-dir', type=str, default='runs/logs',
                        help='Directory for TensorBoard logs (default: runs/logs)')
    parser.add_argument('--tb-image-interval', type=int, default=5,
                        help='Epoch interval for TensorBoard validation prediction images. Set 0 to disable (default: 5)')
    parser.add_argument('--tb-num-images', type=int, default=4,
                        help='Number of validation samples to show in TensorBoard image panels (default: 4)')
    parser.add_argument('--val-tta', type=str, default='none',
                        choices=['none', 'flips'],
                        help='Validation-time test-time augmentation. "flips" averages original/h/v/hv flips (default: none)')
    parser.add_argument('--metric-ignore-classes', type=int, nargs='+', default=None,
                        help='Optional class indices excluded from extra validation metrics only, e.g. --metric-ignore-classes 2')
    
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
    parser.add_argument('--class-weights', type=float, nargs='+', default=None,
                        help='Optional per-output-channel loss weights, e.g. --class-weights 1 1 2 1')
    parser.add_argument('--oversample-class-indices', type=int, nargs='+', default=None,
                        help='Optional mask channel indices to oversample in the training split, e.g. --oversample-class-indices 2')
    parser.add_argument('--oversample-factor', type=float, default=1.0,
                        help='Effective repeat factor for samples containing oversample classes. 1.0 disables it (default: 1.0)')
    parser.add_argument('--oversample-min-pixels', type=int, default=1,
                        help='Minimum positive pixels in a selected class channel to oversample a sample (default: 1)')
    
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
    parser.add_argument('--lr-cosine-t-max', type=int, default=None,
                        help='T_max for CosineAnnealingLR. Defaults to --epochs. Use this to extend training without changing the LR curve.')
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
    try:
        setup_run_outputs(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.class_weights is not None and len(args.class_weights) != args.output_channels:
        raise ValueError(
            f"--class-weights expects {args.output_channels} values, got {len(args.class_weights)}"
        )
    if args.train_patch_size < 0 or (args.train_patch_size > 0 and args.train_patch_size % 32 != 0):
        raise ValueError("--train-patch-size must be 0 or a positive multiple of 32")
    if not 0.0 <= args.patch_positive_probability <= 1.0:
        raise ValueError("--patch-positive-probability must be between 0 and 1")
    if not 0.0 <= args.patch_center_jitter <= 0.5:
        raise ValueError("--patch-center-jitter must be between 0 and 0.5")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.patch_class_indices is not None:
        invalid_patch_classes = [
            idx for idx in args.patch_class_indices
            if idx < 0 or idx >= args.output_channels
        ]
        if invalid_patch_classes:
            raise ValueError(
                f"--patch-class-indices contains invalid values {invalid_patch_classes}; "
                f"valid range is 0..{args.output_channels - 1}"
            )
    
    # Print configuration
    print("=" * 60)
    print("MultiResUNet Training Configuration")
    print("=" * 60)
    print(f"Data Limit: {args.data_limit} samples")
    print(f"Split Mode: {args.split_mode}")
    if args.split_mode == 'random':
        print(f"Validation Split: {args.validation_split:.1%}")
    else:
        print(f"Train Images: {args.train_img_dir}")
        print(f"Train Masks: {args.train_mask_dir}")
        print(f"Validation Images: {args.val_img_dir}")
        print(f"Validation Masks: {args.val_mask_dir}")
        print(f"Test Images: {args.test_img_dir}")
        print(f"Test Masks: {args.test_mask_dir}")
    print(f"Scale Enabled: {args.scale}")
    if args.scale:
        print(f"Scale Factor: {args.scale_factor} ({args.scale_factor*100:.0f}%)")
    print(f"Train Patch Size: {args.train_patch_size if args.train_patch_size > 0 else 'Disabled'}")
    if args.train_patch_size > 0:
        print(f"Patch Positive Probability: {args.patch_positive_probability}")
        print(f"Patch Classes: {args.patch_class_indices if args.patch_class_indices is not None else 'All'}")
        print(f"Patch Min Positive Pixels: {args.patch_min_positive_pixels}")
        print(f"Patch Center Jitter: {args.patch_center_jitter}")
    print(f"Input Channels: {args.input_channels}")
    print(f"Output Channels: {args.output_channels}")
    print(f"Model Architecture: {args.model_architecture}")
    if args.model_architecture == 'smp_unet':
        print(f"Encoder: {args.encoder_name}, weights={args.encoder_weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Evaluation Batch Size: {args.eval_batch_size if args.eval_batch_size is not None else args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gradient Clipping: {args.gradient_clip}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Device: {args.device}")
    print(f"Debug Mode: {args.debug}")
    print(f"Data Validation: {args.check_data}")
    print(f"TensorBoard: {args.tensorboard}")
    if args.tensorboard:
        print(f"Log Directory: {args.log_dir}")
        print(f"TensorBoard Image Interval: {args.tb_image_interval}")
        print(f"TensorBoard Image Samples: {args.tb_num_images}")
    print(f"Run Directory: {args.run_dir}")
    print(f"Model Directory: {args.save_dir}")
    print(f"History Directory: {args.metadata_dir}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Prefetch Factor: {args.prefetch_factor}")
    print(f"Repeat Factor: {args.repeat_factor}")
    print(f"Oversample Classes: {args.oversample_class_indices if args.oversample_class_indices else 'None'}")
    print(f"Oversample Factor: {args.oversample_factor}")
    print(f"Train Augmentation: {args.train_augmentation}")
    print(f"Validation Augmentation: {args.val_augmentation}")
    print(f"Augmentation Strength: {args.augmentation_strength}")
    print(f"Augmentation Curriculum: {args.augmentation_curriculum}")
    if args.augmentation_curriculum != 'none':
        print(f"Curriculum: start=E{args.curriculum_start_epoch}, ramp={args.curriculum_ramp_epochs}, "
              f"max_level={args.curriculum_max_aug_level}, target={args.curriculum_target_strength}")
        if args.augmentation_curriculum == 'adaptive':
            print(f"Adaptive Curriculum: step={args.curriculum_level_step}, "
                  f"window={args.curriculum_adapt_window}, tolerance={args.curriculum_adapt_tolerance}, "
                  f"min_level_epochs={args.curriculum_min_level_epochs}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    print(f"Early Stopping Min Delta: {args.early_stopping_min_delta}")
    print(f"Early Stopping Min Epochs: {args.early_stopping_min_epochs}")
    print(f"Class Weights: {args.class_weights if args.class_weights is not None else 'None'}")
    print(f"Dropout Rate: {args.dropout_rate}")
    print(f"Validation TTA: {args.val_tta}")
    print(f"Final Test Evaluation: {args.run_test_after_training}")
    if args.run_test_after_training:
        print(f"Final Test TTA: {args.test_tta}")
        print(f"Final Test Threshold: {args.test_threshold}")
    print(f"Metric Ignore Classes: {args.metric_ignore_classes if args.metric_ignore_classes else 'None'}")
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
    
    # Option 1: Use memory-efficient Dataset approach.
    if (args.split_mode == 'fixed' or args.data_limit is None or
            args.data_limit > 500 or args.train_patch_size > 0):
        print("Using memory-efficient dataset loading (recommended for large datasets)...")

        if args.split_mode == 'fixed':
            train_dataset, val_dataset, n_train, n_val = create_fixed_datasets(
                train_img_dir=args.train_img_dir,
                train_mask_dir=args.train_mask_dir,
                val_img_dir=args.val_img_dir,
                val_mask_dir=args.val_mask_dir,
                limit=args.data_limit,
                scale=args.scale,
                scale_factor=args.scale_factor,
                original_height=None,
                original_width=None,
                repeat_factor=args.repeat_factor,
                train_apply_augmentation=args.train_augmentation,
                val_apply_augmentation=args.val_augmentation,
                augmentation_strength=args.augmentation_strength,
                strong_aug_strength=args.curriculum_target_strength,
                augmentation_schedule_level=0.0,
                oversample_class_indices=args.oversample_class_indices,
                oversample_factor=args.oversample_factor,
                oversample_min_pixels=args.oversample_min_pixels,
                seed=args.seed,
                train_patch_size=args.train_patch_size,
                patch_positive_probability=args.patch_positive_probability,
                patch_class_indices=args.patch_class_indices,
                patch_min_positive_pixels=args.patch_min_positive_pixels,
                patch_center_jitter=args.patch_center_jitter,
            )
        else:
            train_dataset, val_dataset, n_train, n_val = create_datasets(
                img_dir=IMAGE_DIR,
                mask_dir=MASK_DIR,
                limit=args.data_limit,
                train_ratio=1.0 - args.validation_split,
                scale=args.scale,
                scale_factor=args.scale_factor,
                original_height=None,
                original_width=None,
                repeat_factor=args.repeat_factor,
                train_apply_augmentation=args.train_augmentation,
                val_apply_augmentation=args.val_augmentation,
                augmentation_strength=args.augmentation_strength,
                strong_aug_strength=args.curriculum_target_strength,
                augmentation_schedule_level=0.0,
                oversample_class_indices=args.oversample_class_indices,
                oversample_factor=args.oversample_factor,
                oversample_min_pixels=args.oversample_min_pixels,
                train_patch_size=args.train_patch_size,
                patch_positive_probability=args.patch_positive_probability,
                patch_class_indices=args.patch_class_indices,
                patch_min_positive_pixels=args.patch_min_positive_pixels,
                patch_center_jitter=args.patch_center_jitter,
                shuffle=True,
                seed=args.seed
            )

        test_dataset_for_manifest = None
        if args.split_mode == 'fixed' and os.path.isdir(args.test_img_dir) and os.path.isdir(args.test_mask_dir):
            test_dataset_for_manifest = create_single_dataset(
                img_dir=args.test_img_dir,
                mask_dir=args.test_mask_dir,
                limit=args.test_limit,
                scale=args.scale,
                scale_factor=args.scale_factor,
                apply_augmentation=False,
                augmentation_strength=args.augmentation_strength,
            )
        _save_split_manifest(args, train_dataset, val_dataset, test_dataset_for_manifest)

        train_loader, optimal_workers = _make_loader(train_dataset, args, shuffle=True)
        val_loader, _ = _make_loader(
            val_dataset,
            args,
            shuffle=False,
            batch_size=args.eval_batch_size,
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
        model, model_name = create_model(args)
        model = model.to(device)
        
        print(f"Model architecture: {model_name}")
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
            checkpoint_interval=args.checkpoint_interval,
            verbose=args.verbose,
            log_dir=log_dir,  # Pass TensorBoard log directory
            metadata_dir=args.metadata_dir,
            scale=args.scale,
            scale_factor=args.scale_factor,
            data_limit=args.data_limit,
            validation_split=args.validation_split,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            model_architecture=args.model_architecture,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            train_augmentation=args.train_augmentation,
            val_augmentation=args.val_augmentation,
            repeat_factor=args.repeat_factor,
            train_patch_size=args.train_patch_size,
            patch_positive_probability=args.patch_positive_probability,
            patch_class_indices=args.patch_class_indices,
            patch_min_positive_pixels=args.patch_min_positive_pixels,
            patch_center_jitter=args.patch_center_jitter,
            eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else args.batch_size,
            oversample_class_indices=args.oversample_class_indices,
            oversample_factor=args.oversample_factor,
            oversample_min_pixels=args.oversample_min_pixels,
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
            curriculum_level_step=args.curriculum_level_step,
            curriculum_adapt_window=args.curriculum_adapt_window,
            curriculum_adapt_tolerance=args.curriculum_adapt_tolerance,
            curriculum_min_level_epochs=args.curriculum_min_level_epochs,
            tb_image_interval=args.tb_image_interval,
            tb_num_images=args.tb_num_images,
            val_tta=args.val_tta,
            metric_ignore_classes=args.metric_ignore_classes,
            # Loss function configuration for sparse/imbalanced datasets
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            use_combined_loss=args.use_combined_loss,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            class_weights=args.class_weights,
            # Learning rate scheduler parameters
            lr_scheduler_type=args.lr_scheduler,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            lr_patience=args.lr_patience,
            lr_cosine_t_max=args.lr_cosine_t_max
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
        model, model_name = create_model(args)
        model = model.to(device)
        
        print(f"Model architecture: {model_name}")
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
            checkpoint_interval=args.checkpoint_interval,
            verbose=args.verbose,
            log_dir=log_dir,  # Pass TensorBoard log directory
            metadata_dir=args.metadata_dir,
            scale=args.scale,
            scale_factor=args.scale_factor,
            data_limit=args.data_limit,
            validation_split=args.validation_split,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            model_architecture=args.model_architecture,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
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
            curriculum_level_step=args.curriculum_level_step,
            curriculum_adapt_window=args.curriculum_adapt_window,
            curriculum_adapt_tolerance=args.curriculum_adapt_tolerance,
            curriculum_min_level_epochs=args.curriculum_min_level_epochs,
            tb_image_interval=args.tb_image_interval,
            tb_num_images=args.tb_num_images,
            val_tta=args.val_tta,
            metric_ignore_classes=args.metric_ignore_classes,
            # Loss function configuration for sparse/imbalanced datasets
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            use_combined_loss=args.use_combined_loss,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            class_weights=args.class_weights,
            # Learning rate scheduler parameters
            lr_scheduler_type=args.lr_scheduler,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            lr_patience=args.lr_patience,
            lr_cosine_t_max=args.lr_cosine_t_max
        )
    
    print("-" * 60)
    print("Training complete!")
    
    # Save final model if requested
    if args.save_model:
        print(f"\nSaving final model to {args.save_dir}/")
        saveModel(model, args.save_dir)

    run_final_test_evaluation(args, model, device)

if __name__ == "__main__":
    main()
