import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # predict the mask
            mask_pred = net(image)

            # Handle different mask formats consistently with training:
            # - single-channel class (n_classes==1): expect mask with 0/1 values
            # - multi-class indices (H,W) with values in [0, n_classes-1]
            # - multi-channel binary masks (C,H,W) where C == n_classes
            if net.n_classes == 1:
                # binary mask case
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                # normalize to 0-1 range if needed
                if mask_true.max() > 1.0:
                    mask_true = mask_true / 255.0
                mask_true = (mask_true > 0.5).float()
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # Multi-class case
                if mask_true.dim() == 4 and mask_true.shape[1] == net.n_classes:
                    # Multi-channel binary masks (C,H,W)
                    mask_true = mask_true.to(device=device, dtype=torch.float32)
                    # normalize if needed
                    if mask_true.max() > 1.0:
                        mask_true = mask_true / 255.0
                    # threshold to binary
                    mask_true = (mask_true > 0.5).float()
                    mask_pred_prob = F.sigmoid(mask_pred).float()  # Use sigmoid for multi-channel
                    # compute Dice score, ignoring background channel if present
                    if net.n_classes > 1:
                        dice_score += multiclass_dice_coeff(mask_pred_prob[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                    else:
                        dice_score += multiclass_dice_coeff(mask_pred_prob, mask_true, reduce_batch_first=False)
                else:
                    # Class indices format (H,W) with values [0, n_classes-1]
                    mask_true = mask_true.to(device=device, dtype=torch.long)
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                        f'True mask indices should be in [0, {net.n_classes}[ but got [{mask_true.min()}, {mask_true.max()}]'
                    # convert to one-hot format
                    mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred_onehot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute Dice score, ignoring background
                    if net.n_classes > 1:
                        dice_score += multiclass_dice_coeff(mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:], reduce_batch_first=False)
                    else:
                        dice_score += multiclass_dice_coeff(mask_pred_onehot, mask_true_onehot, reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)