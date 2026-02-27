import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0.0

    # Iterate over the validation set
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            
            # Handle mask format consistently with training
            if net.n_classes == 1:
                # Binary case
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                if mask_true.max() > 1.0:
                    mask_true = mask_true / 255.0
            elif hasattr(dataloader.dataset, 'mask_channels') and dataloader.dataset.mask_channels:
                # Multi-channel binary case
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                if mask_true.max() > 1.0:
                    mask_true = mask_true / 255.0
            else:
                # Multi-class with indices
                mask_true = mask_true.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                mask_pred = net(image)
                
                if net.n_classes == 1:
                    # Binary segmentation
                    mask_pred = torch.sigmoid(mask_pred)
                    dice_score += dice_coeff(mask_pred.squeeze(1), mask_true.float(), reduce_batch_first=False)
                elif hasattr(dataloader.dataset, 'mask_channels') and dataloader.dataset.mask_channels:
                    # Multi-channel binary segmentation
                    mask_pred = torch.sigmoid(mask_pred)
                    dice_score += multiclass_dice_coeff(mask_pred, mask_true.float(), reduce_batch_first=False)
                else:
                    # Multi-class segmentation
                    mask_pred = F.softmax(mask_pred, dim=1)
                    # Convert indices to one-hot for dice calculation
                    mask_true_onehot = F.one_hot(mask_true.squeeze(1), net.n_classes).permute(0, 3, 1, 2).float()
                    dice_score += multiclass_dice_coeff(mask_pred, mask_true_onehot, reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)