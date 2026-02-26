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


def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0.0

    # Iterate over the validation set
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
            image, mask_true = batch['image'], batch['mask']

            # Move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)  # Ensure float type

            with torch.cuda.amp.autocast(enabled=amp, dtype=torch.bfloat16):
                mask_pred = net(image)
                mask_pred = torch.sigmoid(mask_pred)  # Convert logits to probabilities
                dice_score += utils.multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


def train_model(
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
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
):
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    best_score = 0.0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                if true_masks.max() > 1.0:
                    true_masks = true_masks / 255.0

                assert images.shape[1] == model.n_channels, f'Network has been defined with {model.n_channels} input channels, but loaded images have {images.shape[1]} channels.'
                assert true_masks.shape[1] == model.n_classes, f'Network has been defined with {model.n_classes} output channels, but loaded masks have {true_masks.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp, dtype=torch.bfloat16):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
                scaler.step(optimizer)
                scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        val_score = evaluate(model, val_loader, device, amp)
        logging.info(f'Epoch {epoch} finished! Train Loss: {epoch_loss / len(train_loader):.4f}, Validation Dice score: {val_score}')

        if val_score > best_score:
            best_score = val_score
            logging.info(f'New best Dice score: {best_score:.4f}. Saving checkpoint...')
            if save_checkpoint:
                try:
                    os.mkdir('checkpoints')
                    logging.info('Created checkpoints directory.')
                except OSError:
                    pass
                torch.save(model.state_dict(), str('checkpoints/best_model.pth'))
                logging.info(f'Checkpoint saved: best_model.pth with Dice {best_score:.4f}')

        if save_checkpoint:
            torch.save(model.state_dict(), str(f'checkpoints/checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


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
