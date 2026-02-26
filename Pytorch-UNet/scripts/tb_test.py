import os
import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from utils.data_loading import BasicDataset
from torch.utils.tensorboard import SummaryWriter
import logging
import torch

logging.basicConfig(level=logging.INFO)

def main():
    d = BasicDataset('data/imgs','data/masks',scale=0.5)
    try:
        sample = d[0]
        img = sample['image']
        mask = sample['mask']
        print('Loaded sample shapes:', getattr(img,'shape',None), getattr(mask,'shape',None))
        w = SummaryWriter(log_dir='runs/test_local')
        # write scalars and images
        if isinstance(img, torch.Tensor):
            w.add_scalar('test/sample_image_channels', img.shape[0] if len(img.shape)>0 else 0, 0)
            try:
                w.add_image('test/image', img, 0)
            except Exception as e:
                print('image write skipped:', e)
        else:
            try:
                import numpy as np
                arr = img
                if isinstance(arr, np.ndarray) and arr.ndim==3:
                    t = torch.tensor(arr)
                    w.add_image('test/image', t, 0)
            except Exception as e:
                print('image write skipped:', e)
        if isinstance(mask, torch.Tensor):
            w.add_scalar('test/sample_mask_dim', len(mask.shape), 0)
            try:
                if mask.dim()==3:
                    w.add_image('test/mask_channels', mask, 0)
                else:
                    w.add_image('test/mask', mask.unsqueeze(0).float(), 0)
            except Exception as e:
                print('mask write skipped:', e)
        else:
            try:
                import numpy as np
                arr = mask
                if isinstance(arr, np.ndarray):
                    t = torch.tensor(arr)
                    if t.dim()==3:
                        w.add_image('test/mask_channels', t, 0)
                    else:
                        w.add_image('test/mask', t.unsqueeze(0).float(), 0)
            except Exception as e:
                print('mask write skipped:', e)
        w.close()
        print('TensorBoard test written to runs/test_local')
    except Exception as e:
        print('Test load failed:', e)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
