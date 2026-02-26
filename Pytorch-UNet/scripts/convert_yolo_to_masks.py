#!/usr/bin/env python3
"""
Convert YOLO-format bbox labels (.txt) to pixel masks for U-Net.

Usage example:
  python scripts/convert_yolo_to_masks.py --images data/imgs --labels data/labels --output data/masks --multi-class --overwrite

The script tries several heuristics to match a label file to its image
filename (handles names like "name_jpg.rf.hash.txt" -> "name.jpg").
"""
import os
import argparse
import glob
import re
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import sys
import os

# Try to load project-level settings (settings.py at repo root). If it's not
# available, `SETTINGS_PALETTE` will remain None and the script will fall back
# to the built-in default palette.
SETTINGS_PALETTE = None
SETTINGS_CLASS_COLORS = None
try:
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    import settings as project_settings
    SETTINGS_PALETTE = getattr(project_settings, 'PALETTE', None)
    SETTINGS_CLASS_COLORS = getattr(project_settings, 'CLASS_COLORS', None)
except Exception:
    SETTINGS_PALETTE = None
    SETTINGS_CLASS_COLORS = None


def find_image_for_label(label_path, images_dir, exts=("jpg", "jpeg", "png", "bmp")):
    name = os.path.basename(label_path)
    stem = name[:-4] if name.endswith('.txt') else name
    candidates = []
    # common patterns seen in some datasets
    candidates.append(stem.replace('_jpg.rf.', '.jpg'))
    candidates.append(stem.replace('.jpg.rf.', '.jpg'))
    # direct additions
    for ext in exts:
        candidates.append(stem + '.' + ext)
    # try chopping suffixes (before _jpg or .jpg etc)
    m = re.split(r'(_jpg|\.jpg|_jpeg|\.png|_rf)', stem)[0]
    if m:
        for ext in exts:
            candidates.append(m + '.' + ext)
    # check candidates
    for c in candidates:
        p = os.path.join(images_dir, c)
        if os.path.exists(p):
            return p
    # fallback: any file that contains the stem
    for path in glob.glob(os.path.join(images_dir, '*' + stem + '*')):
        return path
    return None


def convert_one(label_path, images_dir, output_dir, multiclass=False, overwrite=False, value_step=1, binary_per_class=False, four_channel=False, channel_map=None, save_npz=False, save_tiff=False, channel_colors=None):
    img_path = find_image_for_label(label_path, images_dir)
    if img_path is None:
        return False, 'no image found'
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    # read label lines
    with open(label_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    if four_channel:
        # create a 4-channel binary image (RGBA) where each channel is a per-class mask
        # channel_map: list of class ids mapped to channels 0..3
        if channel_map is None:
            channel_map = list(range(4))
        # build reverse map: class_id -> channel_index
        class_to_channel = {int(c): i for i, c in enumerate(channel_map)}
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            if cls not in class_to_channel:
                # ignore classes not in mapping
                continue
            ch = class_to_channel[cls]
            x1 = int((xc - w / 2.0) * W)
            y1 = int((yc - h / 2.0) * H)
            x2 = int((xc + w / 2.0) * W)
            y2 = int((yc + h / 2.0) * H)
            # clip
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            # set channel to 255 (binary); use maximum to combine overlaps
            arr[y1:y2, x1:x2, ch] = np.maximum(arr[y1:y2, x1:x2, ch], 255)

        out_path = os.path.join(output_dir, base + '_4ch.png')
        if os.path.exists(out_path) and not overwrite:
            return False, 'exists'
        os.makedirs(output_dir, exist_ok=True)
        img4 = Image.fromarray(arr, mode='RGBA')
        img4.save(out_path)

        # optionally save npz/tiff
        saved_paths = [out_path]
        if save_npz:
            npz_path = os.path.join(output_dir, base + '_4ch.npz')
            # transpose to (C,H,W)
            np.savez_compressed(npz_path, masks=arr.transpose(2, 0, 1))
            saved_paths.append(npz_path)
        if save_tiff:
            # try to use tifffile if available for robust multi-channel TIFF
            try:
                import tifffile
                tiff_path = os.path.join(output_dir, base + '_4ch.tiff')
                # tifffile expects (C,H,W) or (H,W,C) depending; write as (H,W,C)
                tifffile.imwrite(tiff_path, arr)
                saved_paths.append(tiff_path)
            except Exception:
                # fallback: PIL can save RGBA TIFF
                tiff_path = os.path.join(output_dir, base + '_4ch.tiff')
                img4.save(tiff_path)
                saved_paths.append(tiff_path)

        # optionally create a colorized visualization mapping channels->RGB
        if channel_colors:
            # channel_colors: list of (r,g,b) tuples
            # accumulate as int to avoid clipping during sum
            col_acc = np.zeros((H, W, 3), dtype=np.int32)
            for i, col in enumerate(channel_colors):
                if i >= arr.shape[2]:
                    break
                # mask normalized 0/1
                mask_chan = (arr[..., i].astype(np.int32) // 255)
                col_arr = np.array(col, dtype=np.int32).reshape(1, 1, 3)
                col_acc += mask_chan[:, :, None] * col_arr
            col_acc = np.clip(col_acc, 0, 255).astype(np.uint8)
            color_img_path = os.path.join(output_dir, base + '_4ch_color.png')
            Image.fromarray(col_acc, mode='RGB').save(color_img_path)
            saved_paths.append(color_img_path)

        return True, saved_paths if len(saved_paths) > 1 else out_path

    if multiclass and binary_per_class:
        # create one pure B/W mask per class (0 or 255)
        class_masks = {}
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            x1 = int((xc - w / 2.0) * W)
            y1 = int((yc - h / 2.0) * H)
            x2 = int((xc + w / 2.0) * W)
            y2 = int((yc + h / 2.0) * H)
            # clip
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            if cls not in class_masks:
                class_masks[cls] = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(class_masks[cls])
            draw.rectangle([x1, y1, x2, y2], fill=1)

        out_paths = []
        # check existence first
        for cls in class_masks:
            out_path = os.path.join(output_dir, f"{base}_class{cls}.png")
            if os.path.exists(out_path) and not overwrite:
                return False, 'exists'
            out_paths.append(out_path)

        for cls, mask_img in class_masks.items():
            out_path = os.path.join(output_dir, f"{base}_class{cls}.png")
            mask_img.save(out_path)

        saved_paths = out_paths.copy()
        # optionally save combined NPZ/TIFF stacking channels by sorted class id
        if save_npz:
            classes = sorted(class_masks.keys())
            arr_stack = np.stack([np.array(class_masks[c], dtype=np.uint8) for c in classes], axis=0)
            npz_path = os.path.join(output_dir, base + '_classes.npz')
            np.savez_compressed(npz_path, masks=arr_stack)
            saved_paths.append(npz_path)
        if save_tiff:
            try:
                import tifffile
                tiff_path = os.path.join(output_dir, base + '_classes.tiff')
                # arr_stack shape (C,H,W) -> transpose to (H,W,C)
                tifffile.imwrite(tiff_path, arr_stack.transpose(1,2,0))
                saved_paths.append(tiff_path)
            except Exception:
                # if only <=4 channels, we can pack into RGBA
                classes = sorted(class_masks.keys())
                if len(classes) <= 4:
                    pack = np.zeros((H, W, 4), dtype=np.uint8)
                    for i, c in enumerate(classes):
                        pack[..., i] = np.array(class_masks[c], dtype=np.uint8)
                    img4 = Image.fromarray(pack, mode='RGBA')
                    tiff_path = os.path.join(output_dir, base + '_classes.tiff')
                    img4.save(tiff_path)
                    saved_paths.append(tiff_path)
                else:
                    print('tifffile not available and >4 class masks cannot be saved as TIFF without tifffile')

        return True, saved_paths

    # single mask (either single-class or multi-class values)
    mask_arr = np.zeros((H, W), dtype=np.uint8)
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except Exception:
            continue
        x1 = int((xc - w / 2.0) * W)
        y1 = int((yc - h / 2.0) * H)
        x2 = int((xc + w / 2.0) * W)
        y2 = int((yc + h / 2.0) * H)
        # clip
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if multiclass:
            # reserve 0 for background; store (class_id+1)*value_step
            fill = min(255, (cls + 1) * int(value_step))
        else:
            fill = 255
        # set pixels to the maximum value so multiple boxes/classes combine
        mask_arr[y1:y2, x1:x2] = np.maximum(mask_arr[y1:y2, x1:x2], fill)

    mask = Image.fromarray(mask_arr, mode='L')

    out_path = os.path.join(output_dir, base + '.png')
    if os.path.exists(out_path) and not overwrite:
        return False, 'exists'
    mask.save(out_path)

    saved = [out_path]
    # optionally save npz (single-channel -> shape (1,H,W))
    if save_npz:
        npz_path = os.path.join(output_dir, base + '.npz')
        np.savez_compressed(npz_path, masks=mask_arr[np.newaxis, ...])
        saved.append(npz_path)
    if save_tiff:
        try:
            import tifffile
            tiff_path = os.path.join(output_dir, base + '.tiff')
            tifffile.imwrite(tiff_path, mask_arr)
            saved.append(tiff_path)
        except Exception:
            # fallback: save as single-channel TIFF via PIL
            tiff_path = os.path.join(output_dir, base + '.tiff')
            mask.save(tiff_path)
            saved.append(tiff_path)

    return True, saved if len(saved) > 1 else out_path


def main():
    p = argparse.ArgumentParser(description='Convert YOLO .txt to pixel masks')
    p.add_argument('--images', required=True, help='images directory')
    p.add_argument('--labels', required=True, help='labels (.txt) directory')
    p.add_argument('--output', required=True, help='output masks directory')
    p.add_argument('--multi-class', action='store_true', help='produce multi-class masks (pixel value = class_id+1*step)')
    p.add_argument('--value-step', type=int, default=1, help='pixel value step for each class (default=1). class value = (class_id+1)*step')
    p.add_argument('--colorize', action='store_true', help='also save a colorized RGB visualization of the mask')
    p.add_argument('--colors', type=str, default=None, help='optional semicolon-separated RGB colors for classes, e.g. "255,0,0;0,255,0;0,0,255;255,255,0"')
    p.add_argument('--binary-per-class', action='store_true', help='save one pure black/white mask per class (outputs one file per class)')
    p.add_argument('--four-channel', action='store_true', help='produce a single 4-channel (RGBA) PNG where each channel is a binary mask for a class')
    p.add_argument('--channel-map', type=str, default=None, help='comma-separated class ids mapped to channels 0..3 (e.g. "0,1,2,3"). Default: "0,1,2,3"')
    p.add_argument('--channel-colors', type=str, default=None, help='semicolon-separated RGB colors for channels, e.g. "255,0,0;0,255,0;0,0,255;255,255,0"')
    p.add_argument('--overwrite', action='store_true', help='overwrite existing masks')
    p.add_argument('--save-npz', action='store_true', help='also save masks as compressed .npz (arrays saved as uint8, shape=(C,H,W) or (1,H,W))')
    p.add_argument('--save-tiff', action='store_true', help='also save masks as multi-channel TIFF (requires tifffile for >4 channels)')
    args = p.parse_args()

    label_files = sorted([os.path.join(args.labels, f) for f in os.listdir(args.labels) if f.endswith('.txt')])
    if not label_files:
        print('No .txt label files found in', args.labels)
        return
    total = 0
    skipped = 0
    failed = 0
    # prepare colors if needed
    palette = None
    if args.colorize:
        if args.colors:
            try:
                palette = [tuple(map(int, c.split(','))) for c in args.colors.split(';')]
            except Exception:
                print('Invalid --colors format. Expect "r,g,b;..."')
                return
        else:
            # prefer project-level palette if present
            if SETTINGS_PALETTE:
                palette = SETTINGS_PALETTE
            else:
                # default palette for up to 8 classes
                palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,0),(128,0,128)]

    for lab in label_files:
        total += 1
        # prepare channel_map list if provided
        channel_map_list = None
        if args.channel_map:
            try:
                channel_map_list = [int(x) for x in args.channel_map.split(',')]
            except Exception:
                print('Invalid --channel-map format. Expect comma-separated integers like "0,1,2,3"')
                failed += 1
                continue

        # parse channel colors if provided
        channel_colors_list = None
        if args.channel_colors:
            try:
                channel_colors_list = [tuple(map(int, c.split(','))) for c in args.channel_colors.split(';')]
            except Exception:
                print('Invalid --channel-colors format. Expect "r,g,b;..."')
                failed += 1
                continue

        ok, info = convert_one(
            lab,
            args.images,
            args.output,
            multiclass=args.multi_class,
            overwrite=args.overwrite,
            value_step=args.value_step,
            binary_per_class=args.binary_per_class,
            four_channel=args.four_channel,
            channel_map=channel_map_list,
            save_npz=args.save_npz,
            save_tiff=args.save_tiff,
            channel_colors=channel_colors_list,
        )
        if not ok:
            if info == 'exists':
                skipped += 1
            else:
                failed += 1
                print('Failed:', lab, '->', info)
        else:
            # optionally write colorized visualization
            if args.binary_per_class and args.colorize:
                print('Warning: --colorize is not supported with --binary-per-class; skipping colorization for', lab)
            elif args.colorize and args.multi_class and not args.binary_per_class and not args.four_channel:
                base = os.path.splitext(os.path.basename(info))[0]
                mask_path = info
                try:
                    mask = Image.open(mask_path).convert('L')
                    W, H = mask.size
                    color_img = Image.new('RGB', (W, H), (0,0,0))
                    mask_data = mask.load()
                    color_px = color_img.load()
                    # map mask pixel -> class index via value_step
                    for y in range(H):
                        for x in range(W):
                            v = mask_data[x,y]
                            if v == 0:
                                continue
                            idx = (v // args.value_step) - 1 if args.value_step>0 else v-1
                            if idx < 0:
                                continue
                            col = palette[idx] if idx < len(palette) else palette[idx % len(palette)]
                            color_px[x,y] = col
                    color_out = os.path.join(args.output, base + '_color.png')
                    color_img.save(color_out)
                except Exception as e:
                    print('Failed to create colorized mask for', info, '->', e)
    print(f'Done. processed={total} skipped={skipped} failed={failed}')


if __name__ == '__main__':
    main()
