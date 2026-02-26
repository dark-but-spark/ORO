import argparse
import logging
import os
import platform
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

import wandb
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# Optional Weights & Biases support; proceed if available
try:
    import wandb
except Exception:
    wandb = None


def detect_mask_channels(mask_dir: Path):
    """Detect number of channels in mask files under mask_dir.

    Returns int number of channels (1 for single-channel masks), or None if unknown.
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return None
    # find first mask file
    for p in mask_dir.iterdir():
        if p.name.startswith('.'):
            continue
        if p.is_file():
            try:
                if p.suffix == '.npz':
                    data = np.load(p)
                    # prefer 'masks' key
                    if 'masks' in data:
                        arr = data['masks']
                    else:
                        arr = data[data.files[0]]
                    if isinstance(arr, np.ndarray) and arr.ndim == 3:
                        # arr could be (C,H,W) or (H,W,C)
                        if arr.shape[0] <= 8 and arr.shape[0] > 1:
                            return int(arr.shape[0])
                        elif arr.shape[2] <= 8:
                            return int(arr.shape[2])
                        else:
                            return 1
                    elif isinstance(arr, np.ndarray) and arr.ndim == 2:
                        return 1
                else:
                    from PIL import Image
                    im = Image.open(p)
                    mode = im.mode
                    if mode == 'L':
                        return 1
                    elif mode == 'RGB':
                        # RGB masks are unusual; treat as single-channel label
                        return 1
                    elif mode == 'RGBA':
                        return 4
                    else:
                        return 1
            except Exception:
                continue
    return None

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        subset_ratio: float = 1.0,
        mask_channels: int = None,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # Optionally sample a subset of the dataset for quick/fast validation
    if subset_ratio is not None and 0 < subset_ratio < 1.0:
        orig_dataset = dataset
        total = len(dataset)
        subset_len = max(1, int(total * subset_ratio))
        random.seed(0)
        indices = random.sample(range(total), subset_len)
        dataset = Subset(dataset, indices)
        # preserve attributes used later (mask_values, mask_channels)
        if hasattr(orig_dataset, 'mask_values'):
            dataset.mask_values = orig_dataset.mask_values
        if hasattr(orig_dataset, 'mask_channels'):
            dataset.mask_channels = orig_dataset.mask_channels
        logging.info(f'Using subset of dataset: {subset_len}/{total} samples ({subset_ratio*100:.2f}%)')

    # Quick sample check: load first sample and print shapes to confirm pipeline
    try:
        sample = dataset[0]
        img_sample = sample['image']
        mask_sample = sample['mask']
        # tensors may be torch tensors or numpy arrays depending on dataset; convert to shape strings
        img_shape = tuple(img_sample.shape) if hasattr(img_sample, 'shape') else None
        mask_shape = tuple(mask_sample.shape) if hasattr(mask_sample, 'shape') else None
        logging.info(f'Sample loaded: image shape={img_shape}, mask shape={mask_shape}')
    except Exception as e:
        logging.warning(f'Could not load sample for quick check: {e}')

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders (choose safe number of workers; Windows can error if too many)
    cpu_count = os.cpu_count() or 1
    if platform.system() == 'Windows':
        num_workers = min(2, cpu_count)
    else:
        num_workers = min(8, cpu_count)

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    try:
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    except Exception as e:
        logging.warning(f'DataLoader with num_workers={num_workers} failed: {e}. Falling back to num_workers=0')
        loader_args['num_workers'] = 0
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging via TensorBoard SummaryWriter)
    writer = SummaryWriter(log_dir='runs/exp_local')
    # record config
    try:
        writer.add_text('config', str(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                           val_percent=val_percent, save_checkpoint=save_checkpoint,
                                           img_scale=img_scale, amp=amp)), global_step=0)
    except Exception:
        pass

    # Ensure SummaryWriter is closed on exit (bind writer into the callback)
    try:
        import atexit
        atexit.register(lambda w=writer: w.close())
    except Exception:
        pass

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # If using multi-channel binary masks (mask_channels provided), use BCEWithLogitsLoss
    if mask_channels and mask_channels > 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # If multi-channel binary masks, `mask` is float (C,H,W). Otherwise long class indices.
                if mask_channels and mask_channels > 1:
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                    # normalize masks: some datasets store masks as 0/255; convert to 0/1
                    try:
                        if true_masks.max() > 1:
                            true_masks = (true_masks > 1).float()
                            logging.debug('Normalized multi-channel masks from [0,255] to [0,1]')
                    except Exception:
                        pass
                else:
                    # expected class indices or single-channel binary masks
                    # if model expects a single output (n_classes==1) treat masks as binary floats
                    if model.n_classes == 1:
                        true_masks = true_masks.to(device=device, dtype=torch.float32)
                        try:
                            if true_masks.max() > 1:
                                true_masks = (true_masks > 1).float()
                                logging.debug('Normalized single-channel masks from [0,255] to [0,1]')
                        except Exception:
                            pass
                    else:
                        true_masks = true_masks.to(device=device, dtype=torch.long)
                        # If indices exceed expected classes, try to recover binary 0/1 mapping
                        try:
                            if true_masks.max() >= model.n_classes:
                                logging.debug('Mask indices exceed n_classes; thresholding to binary indices')
                                true_masks = (true_masks > 1).long()
                        except Exception:
                            pass

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if mask_channels and mask_channels > 1:
                        # multi-channel binary: BCE per channel + multiclass dice over channels
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.sigmoid(masks_pred).float(), true_masks.float(), multiclass=True)
                    else:
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            # Ensure true_masks is Long for CrossEntropyLoss
                            true_masks = true_masks.to(device=device, dtype=torch.long)
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                optimizer.zero_grad(set_to_none=True)
                # If loss is non-finite (NaN/Inf) skip backward/update to avoid corrupting weights
                if not torch.isfinite(loss):
                    logging.warning(f'Non-finite loss detected at step {global_step}. Skipping backward/update.')
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                try:
                    writer.add_scalar('train/loss', loss.item(), global_step)
                except Exception:
                    pass
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            try:
                                if wandb is not None:
                                    if not (torch.isinf(value) | torch.isnan(value)).any():
                                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                            except Exception:
                                pass

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            writer.add_scalar('val/dice', val_score, global_step)
                            writer.add_scalar('val/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                            # log an example image and masks (first in batch)
                            try:
                                img_vis = images[0].cpu()
                                # images are C,H,W in 0..1
                                writer.add_image('val/image', img_vis, global_step)
                            except Exception:
                                pass
                            try:
                                # true_masks may be long or float
                                if true_masks.dim() == 3:
                                    # (C,H,W) -> grid
                                    writer.add_image('val/mask_true_channels', true_masks[0].cpu(), global_step)
                                else:
                                    writer.add_image('val/mask_true', true_masks[0].unsqueeze(0).float().cpu(), global_step)
                            except Exception:
                                pass
                            try:
                                pred = masks_pred.argmax(dim=1)[0].float().cpu()
                                writer.add_image('val/mask_pred', pred.unsqueeze(0), global_step)
                            except Exception:
                                pass
                            # log histograms
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                try:
                                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                                    if value.grad is not None:
                                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                                except Exception:
                                    pass
                        except Exception:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            # save detected mask channels if available
            mask_ch = None
            if hasattr(dataset, 'mask_channels') and dataset.mask_channels is not None:
                mask_ch = int(dataset.mask_channels)
            elif mask_channels is not None:
                mask_ch = int(mask_channels)
            state_dict['mask_channels'] = mask_ch
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        # flush writer periodically
        try:
            writer.flush()
        except Exception:
            pass


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--mask-channels', type=int, default=None, help='If provided, treat masks as multi-channel binary with this many channels (e.g. 4)')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of dataset to use for training/validation (0-1], default 1 uses all)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Auto-detect mask channels if not provided
    detected = None
    if args.mask_channels is None:
        try:
            detected = detect_mask_channels(dir_mask)
        except Exception:
            detected = None

    if args.mask_channels is None and detected is not None:
        logging.info(f'Auto-detected mask channels: {detected}')
        n_classes = detected
    else:
        n_classes = args.mask_channels if args.mask_channels is not None else args.classes

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    # SummaryWriter is closed inside train_model when created; no global registration needed

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            subset_ratio=args.subset,
            mask_channels=args.mask_channels
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
