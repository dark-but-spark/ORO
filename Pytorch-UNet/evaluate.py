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

            # Handle different mask formats:
            # - single-channel class (n_classes==1): expect mask with 0/1 values
            # - multi-class indices (H,W) with values in [0, n_classes-1]
            # - multi-channel binary masks (C,H,W) where C == n_classes
            if net.n_classes == 1:
                # binary mask expected
                mask_true = mask_true.to(device=device, dtype=torch.float32)
                # ensure binary
                mask_true = (mask_true > 0.5).float()
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # If mask comes as multi-channel (batch, C, H, W) and C == n_classes,
                # treat it as binary per-channel masks (possibly 0/255) and convert.
                if mask_true.dim() == 4 and mask_true.shape[1] == net.n_classes:
                    mask_true = mask_true.to(device=device, dtype=torch.float32)
                    # threshold any non-zero values to 1.0
                    mask_true = (mask_true > 0.5).float()
                    mask_pred_prob = F.softmax(mask_pred, dim=1).float()
                    # compute Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred_prob[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                else:
                    # assume mask_true contains class indices
                    mask_true = mask_true.to(device=device, dtype=torch.long)
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
