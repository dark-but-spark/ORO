import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext == '.npz':
        # return raw numpy array for .npz masks (expects key 'masks' or first array)
        data = np.load(filename)
        if 'masks' in data:
            return data['masks']
        # pick first array in archive
        first = data.files[0]
        return data[first]
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Collect image ids but keep only those that have a corresponding mask file
        all_image_files = [f for f in listdir(images_dir) if isfile(join(images_dir, f)) and not f.startswith('.')]
        all_ids = [splitext(file)[0] for file in all_image_files]

        # gather mask basenames for prefix matching (handles extra suffixes like '_4ch')
        mask_files = [f for f in listdir(self.mask_dir) if isfile(join(self.mask_dir, f)) and not f.startswith('.')]
        mask_basenames = [splitext(f)[0] for f in mask_files]

        ids_with_masks = []
        missing_masks = []
        for iid in all_ids:
            # exact match using provided mask_suffix
            matches = [mb for mb in mask_basenames if mb == iid + mask_suffix]
            # fallback: any mask file whose basename startswith the image id
            if not matches:
                matches = [mb for mb in mask_basenames if mb.startswith(iid)]

            if matches:
                ids_with_masks.append(iid)
            else:
                missing_masks.append(iid)

        if not ids_with_masks:
            raise RuntimeError(f'No input file with matching mask found in {images_dir} / {mask_dir}')

        if missing_masks:
            logging.warning(f'Skipping {len(missing_masks)} images without masks (examples: {missing_masks[:5]})')

        self.ids = ids_with_masks

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Detect if masks are stored as .npz multi-channel arrays. If so, skip
        # the unique value scan and treat masks as multi-channel binary/float arrays.
        def _find_mask_files_for_id(iid):
            exact = list(self.mask_dir.glob(iid + self.mask_suffix + '.*'))
            if exact:
                return exact
            # fallback: any mask file whose name starts with the id
            return [p for p in self.mask_dir.iterdir() if p.is_file() and p.name.startswith(iid)]

        sample_mask_files = _find_mask_files_for_id(self.ids[0])
        self.multi_channel_masks = False
        self.mask_channels = None
        if sample_mask_files:
            if sample_mask_files[0].suffix == '.npz':
                # inspect first mask to determine channels
                arr = load_image(sample_mask_files[0])
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    # expect shape (C,H,W) or (H,W,C)
                    if arr.shape[0] <= 8 and arr.shape[0] > 1:
                        self.multi_channel_masks = True
                        # assume (C,H,W)
                        self.mask_channels = int(arr.shape[0])

        if self.multi_channel_masks:
            logging.info(f'Detected multi-channel masks (npz) with {self.mask_channels} channels')
            # mask_values not used for multi-channel masks
            self.mask_values = None
        else:
            logging.info('Scanning mask files to determine unique values')
            with Pool() as p:
                unique = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                    total=len(self.ids)
                ))

            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
            logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    def _find_files_for_id(self, directory: Path, iid: str, suffix: str = ''):
        # exact match first
        exact = list(directory.glob(iid + suffix + '.*'))
        if exact:
            return exact
        # fallback: any file starting with iid (handles extra suffixes like '_4ch')
        return [p for p in directory.iterdir() if p.is_file() and p.name.startswith(iid)]

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # pil_img can be a PIL.Image or a numpy array (for .npz masks)
        if isinstance(pil_img, np.ndarray):
            # numpy array
            arr = pil_img
            if is_mask:
                # arr may be (C,H,W) or (H,W,C) or (H,W)
                if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[0] > 1:
                    # assume (C,H,W) -> transpose to (H,W,C)
                    arr_hw = arr.transpose(1, 2, 0)
                elif arr.ndim == 3 and arr.shape[2] <= 8:
                    arr_hw = arr
                else:
                    arr_hw = arr

                # resize per-channel using PIL nearest
                H0, W0 = arr_hw.shape[0], arr_hw.shape[1]
                newW, newH = int(scale * W0), int(scale * H0)
                assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
                if arr_hw.ndim == 2:
                    pil = Image.fromarray(arr_hw)
                    pil = pil.resize((newW, newH), resample=Image.NEAREST)
                    out = np.asarray(pil)
                    return out
                else:
                    C = arr_hw.shape[2]
                    channels = []
                    for c in range(C):
                        pil = Image.fromarray(arr_hw[:, :, c])
                        pil = pil.resize((newW, newH), resample=Image.NEAREST)
                        channels.append(np.asarray(pil))
                    # return as (C, newH, newW)
                    stack = np.stack(channels, axis=0)
                    return stack
            else:
                # image provided as numpy array (H,W,C) or (C,H,W)
                if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[2] > 4:
                    # unlikely; assume (H,W,C)
                    img_hw = arr
                elif arr.ndim == 3 and arr.shape[0] <= 4:
                    # (C,H,W)
                    img_hw = arr.transpose(1, 2, 0)
                else:
                    img_hw = arr
                H0, W0 = img_hw.shape[0], img_hw.shape[1]
                newW, newH = int(scale * W0), int(scale * H0)
                assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
                # resize using PIL BICUBIC per channel if needed
                if img_hw.ndim == 2:
                    pil = Image.fromarray(img_hw)
                    pil = pil.resize((newW, newH), resample=Image.BICUBIC)
                    img = np.asarray(pil)
                else:
                    C = img_hw.shape[2]
                    channels = []
                    for c in range(C):
                        pil = Image.fromarray(img_hw[:, :, c])
                        pil = pil.resize((newW, newH), resample=Image.BICUBIC)
                        channels.append(np.asarray(pil))
                    img = np.stack(channels, axis=2)

                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                else:
                    img = img.transpose((2, 0, 1))

                if (img > 1).any():
                    img = img / 255.0

                return img

        # PIL image path (original behavior)
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # If mask_values is None, assume mask is multi-channel one-hot like (C,H,W)
            if mask_values is None:
                # img may be (H,W,C) or (C,H,W)
                if img.ndim == 3 and img.shape[0] <= 8 and img.shape[2] > 8:
                    # unlikely; convert to (C,H,W)
                    arr = img.transpose(2, 0, 1)
                elif img.ndim == 3 and img.shape[2] <= 8:
                    arr = img.transpose(2, 0, 1) if img.shape[2] <= 8 else img
                elif img.ndim == 3 and img.shape[0] <= 8:
                    arr = img
                else:
                    # fallback: single channel
                    return img

                # return as (C,H,W) with binary 0/1
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                return arr

            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = self._find_files_for_id(self.mask_dir, name, self.mask_suffix)
        img_file = self._find_files_for_id(self.images_dir, name)

        def _select_best(files, preferred_exts):
            if not files:
                return []
            if len(files) == 1:
                return [files[0]]
            # sort by preferred extensions order, unknown ext go last
            pref = [e.lower() for e in preferred_exts]
            files_sorted = sorted(files, key=lambda p: pref.index(p.suffix.lower()) if p.suffix.lower() in pref else len(pref))
            return [files_sorted[0]]

        img_candidates = _select_best(img_file, ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'])
        mask_candidates = _select_best(mask_file, ['.npz', '.pt', '.pth', '.tiff', '.tif', '.png', '.jpg', '.jpeg'])

        if not img_candidates:
            raise AssertionError(f'Either no image found for the ID {name}: {img_file}')
        if not mask_candidates:
            raise AssertionError(f'Either no mask found for the ID {name}: {mask_file}')

        # use the selected best candidate
        mask = load_image(mask_candidates[0])
        img = load_image(img_candidates[0])

        # compare sizes robustly when mask may be a numpy array (from .npz)
        def _size_of(x):
            if isinstance(x, np.ndarray):
                if x.ndim == 2:
                    return (x.shape[1], x.shape[0])
                elif x.ndim == 3:
                    # could be (C,H,W) or (H,W,C)
                    if x.shape[0] <= 8 and x.shape[0] > 1:
                        return (x.shape[2], x.shape[1])
                    else:
                        return (x.shape[1], x.shape[0])
                else:
                    return (x.shape[1], x.shape[0])
            else:
                return x.size

        assert _size_of(img) == _size_of(mask), \
            f'Image and mask {name} should be the same size, but are {_size_of(img)} and {_size_of(mask)}'
        # handle numpy arrays for masks/images
        img_pre = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask_pre = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        if isinstance(mask_pre, np.ndarray) and mask_pre.ndim == 3:
            # multi-channel mask -> return float tensor (C,H,W)
            mask_tensor = torch.as_tensor(mask_pre.copy()).float().contiguous()
        else:
            mask_tensor = torch.as_tensor(mask_pre.copy()).long().contiguous()

        return {
            'image': torch.as_tensor(img_pre.copy()).float().contiguous(),
            'mask': mask_tensor
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
