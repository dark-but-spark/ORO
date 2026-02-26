#!/usr/bin/env python3
"""
Debug helper to inspect one YOLO label, matched image, and generated mask.

Usage:
  python scripts/debug_mask_check.py --label data/train/labels/abc.txt --images data/train/images --masks data/train/masks

It will print image path/size, show parsed YOLO boxes and pixel coords, and
report unique values in the mask (if present). It also writes an overlay image
`<labelbase>_overlay.png` in the current folder for quick visual check.
"""
import os
import argparse
import glob
import re
from PIL import Image, ImageDraw, ImageOps


def find_image_for_label(label_path, images_dir, exts=("jpg", "jpeg", "png", "bmp")):
    name = os.path.basename(label_path)
    stem = name[:-4] if name.endswith('.txt') else name
    candidates = []
    candidates.append(stem.replace('_jpg.rf.', '.jpg'))
    candidates.append(stem.replace('.jpg.rf.', '.jpg'))
    for ext in exts:
        candidates.append(stem + '.' + ext)
    m = re.split(r'(_jpg|\.jpg|_jpeg|\.png|_rf)', stem)[0]
    if m:
        for ext in exts:
            candidates.append(m + '.' + ext)
    for c in candidates:
        p = os.path.join(images_dir, c)
        if os.path.exists(p):
            return p
    for path in glob.glob(os.path.join(images_dir, '*' + stem + '*')):
        return path
    return None


def parse_yolo_lines(lines):
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except Exception:
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--masks', required=True)
    args = p.parse_args()

    if not os.path.exists(args.label):
        print('Label not found:', args.label)
        return
    with open(args.label, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    print('Label lines:', lines)
    boxes = parse_yolo_lines(lines)
    print('Parsed boxes (class, xc, yc, w, h):')
    for b in boxes:
        print(' ', b)

    img_path = find_image_for_label(args.label, args.images)
    if img_path is None:
        print('No matching image found in', args.images)
        return
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    print('Matched image:', img_path)
    print('Image size (W,H):', W, H)

    pixel_boxes = []
    for cls, xc, yc, w, h in boxes:
        x1 = int((xc - w / 2.0) * W)
        y1 = int((yc - h / 2.0) * H)
        x2 = int((xc + w / 2.0) * W)
        y2 = int((yc + h / 2.0) * H)
        pixel_boxes.append((cls, x1, y1, x2, y2))
    print('Pixel boxes (class, x1,y1,x2,y2):')
    for pb in pixel_boxes:
        print(' ', pb)

    # check corresponding mask if exists
    base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(args.masks, base + '.png')
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
        vals = sorted(set(mask.getdata()))
        print('Mask found:', mask_path)
        print('Unique mask values (sample):', vals[:20])
    else:
        print('Mask not found at', mask_path)

    # write overlay for visual check
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    for cls, x1, y1, x2, y2 in pixel_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=3)
    # if mask exists, overlay it semi transparent
    if os.path.exists(mask_path):
        mask_rgb = Image.open(mask_path).convert('L')
        color_mask = ImageOps.colorize(mask_rgb, black=(0,0,0), white=(0,255,0))
        overlay = Image.blend(overlay, color_mask, alpha=0.4)

    outname = base + '_overlay.png'
    overlay.save(outname)
    print('Wrote overlay to', outname)


if __name__ == '__main__':
    main()
