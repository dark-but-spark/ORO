import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(img_dir):
    return sorted(
        path for path in Path(img_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _draw_yolo_row(mask, row, width, height, num_classes):
    parts = row.strip().split()
    if not parts:
        return False

    class_id = int(float(parts[0]))
    if class_id < 0 or class_id >= num_classes:
        return False

    values = [float(value) for value in parts[1:]]
    if len(values) == 4:
        x_center, y_center, bbox_width, bbox_height = values
        x1 = int(round((x_center - bbox_width / 2.0) * width))
        y1 = int(round((y_center - bbox_height / 2.0) * height))
        x2 = int(round((x_center + bbox_width / 2.0) * width))
        y2 = int(round((y_center + bbox_height / 2.0) * height))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2, class_id] = 255
            return True
        return False

    if len(values) >= 6 and len(values) % 2 == 0:
        points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon = np.round(points).astype(np.int32)
        channel_mask = mask[:, :, class_id].copy()
        cv2.fillPoly(channel_mask, [polygon], 255)
        mask[:, :, class_id] = channel_mask
        return True

    return False


def yolo_to_npz(img_dir, label_dir, output_dir, num_classes=4, overwrite=True):
    """
    Convert YOLO bbox or YOLO segmentation polygon labels to multi-channel .npz masks.

    Images and labels are matched by stem. Missing label files produce an all-zero
    mask, which is the correct representation for negative images.
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "img_dir": str(img_dir),
        "label_dir": str(label_dir),
        "output_dir": str(output_dir),
        "num_classes": int(num_classes),
        "images": 0,
        "missing_labels": 0,
        "invalid_images": 0,
        "objects": 0,
        "skipped_rows": 0,
    }

    images = _image_files(img_dir)
    for img_path in tqdm(images, desc=f"Converting {img_dir}"):
        out_path = output_dir / f"{img_path.stem}.npz"
        if out_path.exists() and not overwrite:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["invalid_images"] += 1
            continue

        height, width = image.shape[:2]
        mask = np.zeros((height, width, num_classes), dtype=np.uint8)
        label_path = label_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as label_file:
                for row in label_file:
                    try:
                        drawn = _draw_yolo_row(mask, row, width, height, num_classes)
                    except (ValueError, IndexError):
                        drawn = False
                    if drawn:
                        stats["objects"] += 1
                    elif row.strip():
                        stats["skipped_rows"] += 1
        else:
            stats["missing_labels"] += 1

        np.savez_compressed(out_path, mask=mask)
        stats["images"] += 1

    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(stats, summary_file, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def convert_standard_splits(data_root, splits, num_classes=4, overwrite=True):
    data_root = Path(data_root)
    all_stats = {}
    for split in splits:
        split_dir = data_root / split
        all_stats[split] = yolo_to_npz(
            img_dir=split_dir / "images",
            label_dir=split_dir / "labels",
            output_dir=split_dir / "masks",
            num_classes=num_classes,
            overwrite=overwrite,
        )
    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO labels to .npz segmentation masks.")
    parser.add_argument("--data-root", default="data", help="Dataset root containing train/valid/test folders.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Splits to convert.")
    parser.add_argument("--img-dir", default=None, help="Single image directory. Overrides --data-root/--splits.")
    parser.add_argument("--label-dir", default=None, help="Single YOLO label directory. Required with --img-dir.")
    parser.add_argument("--output-dir", default=None, help="Single output mask directory. Required with --img-dir.")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--no-overwrite", action="store_true", help="Skip masks that already exist.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overwrite = not args.no_overwrite

    if args.img_dir:
        if not args.label_dir or not args.output_dir:
            raise SystemExit("--label-dir and --output-dir are required with --img-dir")
        yolo_to_npz(args.img_dir, args.label_dir, args.output_dir, args.num_classes, overwrite)
    else:
        convert_standard_splits(args.data_root, args.splits, args.num_classes, overwrite)
