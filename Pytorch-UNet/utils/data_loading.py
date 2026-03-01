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
        # For 2D masks, flatten to 1D for consistency
        return np.unique(mask.flatten())
    elif mask.ndim == 3:
        # For 3D masks, reshape appropriately
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

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Detect if masks are stored as .npz multi-channel arrays. If so, skip
        # the unique value scan and treat masks as multi-channel binary/float arrays.
        sample_mask_files = list(self.mask_dir.glob(self.ids[0] + self.mask_suffix + '.*'))
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
                unique_results = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                    total=len(self.ids)
                ))
            
            # Handle mixed dimensions safely - convert all to flat lists
            flattened_unique = []
            for unique_arr in unique_results:
                if unique_arr.ndim == 1:
                    # 1D array (from 2D masks)
                    flattened_unique.extend(unique_arr.tolist())
                else:
                    # 2D array (from 3D masks) - flatten each row
                    for row in unique_arr:
                        flattened_unique.extend(row.tolist())
            
            # Remove duplicates and sort
            self.mask_values = sorted(list(set(flattened_unique)))
            logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    def _find_files_for_id(self, directory: Path, iid: str, suffix: str = ''):
        # Prioritize _4ch.npz files
        exact_4ch = list(directory.glob(iid + '_4ch.npz'))
        if exact_4ch:
            return exact_4ch

        # Fallback to exact match
        exact = list(directory.glob(iid + suffix + '.*'))
        if exact:
            return exact

        # Fallback: any file starting with iid (handles extra suffixes like '_4ch')
        return [p for p in directory.iterdir() if p.is_file() and p.name.startswith(iid)]

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        """Resize the image or mask to the desired scale."""
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, f"Scale {scale} is too small for image size {w}x{h}."

        pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if is_mask:
            # Ensure mask values are in the expected range
            mask = np.zeros((len(mask_values), new_h, new_w), dtype=np.float32)
            for i, v in enumerate(mask_values):
                mask[i, :, :] = (img_ndarray == v).astype(np.float32)
            return mask
        else:
            # Normalize image values to [0, 1] and transpose to channel-first format
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            return img_ndarray / 255

    def preprocess_mask(self, pil_img):
        """
        Ensure masks are preprocessed to have 4 channels (C, H, W).
        Maintains [4,640,640] tensor dimension format as per project specifications.
        """
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        # Resize the mask
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST)

        # Convert to numpy array
        img_ndarray = np.asarray(pil_img)

        # Ensure 4-channel masks maintaining proper dimension order
        if img_ndarray.ndim == 2:  # Single-channel mask
            # Expand to 4 channels: (H, W) -> (4, H, W)
            mask = np.stack([img_ndarray] * 4, axis=0).astype(np.float32)
        elif img_ndarray.ndim == 3:
            if img_ndarray.shape[2] == 4:  # RGBA format
                # Convert RGBA to (C, H, W) format
                mask = img_ndarray.transpose((2, 0, 1)).astype(np.float32)
            elif img_ndarray.shape[0] == 4:  # Already in (C, H, W) format
                mask = img_ndarray.astype(np.float32)
            elif img_ndarray.shape[2] == 1:  # (H, W, 1) format
                # Remove singleton dimension and expand to 4 channels
                single_channel = img_ndarray.squeeze(2)
                mask = np.stack([single_channel] * 4, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unexpected 3D mask shape: {img_ndarray.shape}")
        else:
            raise ValueError(f"Unexpected mask dimensions: {img_ndarray.ndim}D array with shape {img_ndarray.shape}")

        # Verify final shape is (4, H, W)
        assert mask.ndim == 3 and mask.shape[0] == 4, \
            f"Expected 4-channel mask with shape (4, H, W), got {mask.shape}"
            
        return mask

    def __getitem__(self, idx):
        """
        Override to preprocess images and masks.
        """
        name = self.ids[idx]

        # Load and preprocess the mask
        mask_file = self._find_files_for_id(self.mask_dir, name, self.mask_suffix)[0]
        if mask_file.suffix == '.npz':
            # Dynamically check available keys in the .npz file
            npz_data = np.load(mask_file)
            available_keys = list(npz_data.keys())
            logging.info(f"Available keys in {mask_file.name}: {available_keys}")

            # Use the first key if 'arr_0' is not present
            key_to_use = 'arr_0' if 'arr_0' in available_keys else available_keys[0]
            mask = npz_data[key_to_use]
            if mask.ndim == 2:  # Single-channel mask
                mask = np.stack([mask] * 4, axis=0).astype(np.float32)
            elif mask.ndim == 3 and mask.shape[0] != 4:  # Ensure 4 channels
                raise ValueError(f"Expected 4-channel mask, but got shape {mask.shape}")
        else:
            mask = Image.open(mask_file).convert('L')  # Convert to grayscale
            mask = np.stack([np.array(mask)] * 4, axis=0).astype(np.float32)  # Expand to 4 channels

        # Ensure the mask file is a _4ch.npz file
        if '_4ch.npz' not in mask_file.name:
            raise ValueError(f"Expected a '_4ch.npz' file, but got {mask_file.name}")

        # Load and preprocess the image
        img_file = self._find_files_for_id(self.images_dir, name)[0]
        img = Image.open(img_file).convert('RGB')  # Ensure RGB format
        img = self.preprocess(None, img, self.scale, is_mask=False)

        return {'image': img, 'mask': mask}


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
