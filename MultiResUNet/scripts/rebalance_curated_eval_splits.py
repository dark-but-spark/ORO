#!/usr/bin/env python
"""Re-split a curated valid/test pool by original-image group and class.

This utility never edits the source curated dataset. It pools the selected
valid/test samples, keeps all ``.rf.`` variants of one original image together,
and writes a new group-disjoint valid/test dataset.
"""

import argparse
import csv
import itertools
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument(
        "--test-groups-per-class",
        default="0:2,1:7,2:3,3:4",
        help="Exact test group count per class, e.g. 0:2,1:7,2:3,3:4",
    )
    parser.add_argument("--target-test-fraction", type=float, default=0.48)
    return parser.parse_args()


def original_group(stem):
    return stem.split(".rf.", 1)[0]


def parse_targets(value):
    targets = {}
    for item in value.split(","):
        class_idx, count = item.split(":", 1)
        targets[int(class_idx)] = int(count)
    return targets


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_or_link(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def find_image(directory, stem):
    for path in directory.glob(f"{stem}.*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def load_groups(source_root, selected_rows):
    groups = defaultdict(list)
    for row in selected_rows:
        old_split = row["split"].strip()
        stem = row["image_stem"].strip()
        image_path = find_image(source_root / old_split / "images", stem)
        mask_path = source_root / old_split / "masks" / f"{stem}.npz"
        label_path = source_root / old_split / "labels" / f"{stem}.txt"
        if image_path is None or not mask_path.exists():
            raise FileNotFoundError(f"Missing curated pair for {old_split}/{stem}")

        mask = np.load(mask_path)["mask"] > 0
        if mask.ndim != 3:
            raise ValueError(f"Expected HWC mask for {mask_path}, got {mask.shape}")
        class_pixels = mask.reshape(-1, mask.shape[2]).sum(axis=0).astype(np.int64)
        groups[original_group(stem)].append(
            {
                "row": dict(row),
                "old_split": old_split,
                "stem": stem,
                "image_path": image_path,
                "mask_path": mask_path,
                "label_path": label_path if label_path.exists() else None,
                "class_pixels": class_pixels,
            }
        )

    group_records = {}
    output_channels = None
    for group_id, items in groups.items():
        output_channels = len(items[0]["class_pixels"])
        pixels = np.stack([item["class_pixels"] for item in items]).sum(axis=0)
        present = tuple(int(value > 0) for value in pixels)
        present_classes = [idx for idx, value in enumerate(present) if value]
        if len(present_classes) != 1:
            raise ValueError(
                f"Expected one class per curated group, got classes {present_classes} for {group_id}"
            )
        if len({item["old_split"] for item in items}) != 1:
            raise ValueError(f"Source curated split already leaks group {group_id}")
        group_records[group_id] = {
            "group_id": group_id,
            "class_idx": present_classes[0],
            "class_pixels": pixels,
            "image_count": len(items),
            "old_split": items[0]["old_split"],
            "items": items,
        }
    return group_records, output_channels


def select_test_groups(group_records, targets, target_fraction):
    by_class = defaultdict(list)
    for record in group_records.values():
        by_class[record["class_idx"]].append(record)

    selected = set()
    selection_details = {}
    for class_idx in sorted(by_class):
        records = sorted(by_class[class_idx], key=lambda record: record["group_id"])
        requested = targets.get(class_idx, round(len(records) * target_fraction))
        if requested <= 0 or requested >= len(records):
            raise ValueError(
                f"Class {class_idx} needs groups in both splits; requested {requested} of {len(records)}"
            )

        total_pixels = sum(int(record["class_pixels"][class_idx]) for record in records)
        total_images = sum(record["image_count"] for record in records)
        best = None
        for combination in itertools.combinations(records, requested):
            test_pixels = sum(int(record["class_pixels"][class_idx]) for record in combination)
            test_images = sum(record["image_count"] for record in combination)
            pixel_fraction = test_pixels / max(1, total_pixels)
            image_fraction = test_images / max(1, total_images)
            score = abs(pixel_fraction - target_fraction) + 0.35 * abs(image_fraction - target_fraction)
            tie_key = tuple(record["group_id"] for record in combination)
            candidate = (score, tie_key, combination, pixel_fraction, image_fraction)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        _, _, combination, pixel_fraction, image_fraction = best
        class_groups = [record["group_id"] for record in combination]
        selected.update(class_groups)
        selection_details[str(class_idx)] = {
            "available_groups": len(records),
            "test_groups": len(class_groups),
            "test_group_ids": class_groups,
            "test_pixel_fraction": pixel_fraction,
            "test_image_fraction": image_fraction,
        }
    return selected, selection_details


def split_stats(assignments, output_channels):
    stats = {}
    for split in ("valid", "test"):
        records = [record for record in assignments.values() if record["new_split"] == split]
        group_counts = [0] * output_channels
        image_counts = [0] * output_channels
        pixel_counts = [0] * output_channels
        for record in records:
            class_idx = record["class_idx"]
            group_counts[class_idx] += 1
            image_counts[class_idx] += record["image_count"]
            pixel_counts[class_idx] += int(record["class_pixels"][class_idx])
        stats[split] = {
            "groups": len(records),
            "images": sum(record["image_count"] for record in records),
            "class_group_counts": group_counts,
            "class_positive_image_counts": image_counts,
            "class_positive_pixel_counts": pixel_counts,
        }
    return stats


def main():
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {args.output_root}")
    if not 0.0 < args.target_test_fraction < 1.0:
        raise ValueError("--target-test-fraction must be between 0 and 1")

    selected_path = args.source_root / "selected_rows.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected_rows = read_csv(selected_path)
    group_records, output_channels = load_groups(args.source_root, selected_rows)
    targets = parse_targets(args.test_groups_per_class)
    test_groups, selection_details = select_test_groups(
        group_records, targets, args.target_test_fraction
    )

    assignments = {}
    copy_modes = Counter()
    output_rows = []
    for group_id, record in sorted(group_records.items()):
        new_split = "test" if group_id in test_groups else "valid"
        record["new_split"] = new_split
        assignments[group_id] = record
        for item in record["items"]:
            copy_modes[copy_or_link(
                item["image_path"],
                args.output_root / new_split / "images" / item["image_path"].name,
                args.copy_mode,
            )] += 1
            copy_modes[copy_or_link(
                item["mask_path"],
                args.output_root / new_split / "masks" / item["mask_path"].name,
                args.copy_mode,
            )] += 1
            if item["label_path"] is not None:
                copy_modes[copy_or_link(
                    item["label_path"],
                    args.output_root / new_split / "labels" / item["label_path"].name,
                    args.copy_mode,
                )] += 1

            row = dict(item["row"])
            row["original_split"] = item["old_split"]
            row["split"] = new_split
            row["group_id"] = group_id
            row["class_index_from_mask"] = record["class_idx"]
            row["group_class_pixels"] = int(record["class_pixels"][record["class_idx"]])
            output_rows.append(row)

    stats = split_stats(assignments, output_channels)
    for split in ("valid", "test"):
        if any(count == 0 for count in stats[split]["class_group_counts"]):
            raise RuntimeError(f"Incomplete class coverage after split: {split}: {stats[split]}")

    valid_groups = {group_id for group_id, record in assignments.items() if record["new_split"] == "valid"}
    test_groups_check = {group_id for group_id, record in assignments.items() if record["new_split"] == "test"}
    overlap = sorted(valid_groups & test_groups_check)
    if overlap:
        raise RuntimeError(f"Group leakage after split: {overlap}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    write_csv(args.output_root / "selected_rows.csv", output_rows, fieldnames)
    write_csv(args.output_root / "balanced_manifest.csv", output_rows, fieldnames)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "copy_mode": args.copy_mode,
        "target_test_fraction": args.target_test_fraction,
        "requested_test_groups_per_class": targets,
        "source_images": len(selected_rows),
        "source_groups": len(group_records),
        "selection_details": selection_details,
        "split_stats": stats,
        "valid_test_group_overlap": overlap,
        "copy_modes": dict(copy_modes),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        "nc: 4\nnames: ['ballooning', 'fibrosis', 'inflammation', 'steatosis']\n",
        encoding="utf-8",
    )
    (args.output_root / "README.md").write_text(
        "# Balanced curated B evaluation split\n\n"
        "Built from the strict reviewed curated pool. All variants of an original "
        "`.rf.` group stay in one split. Both valid and test contain all four classes. "
        "The source curated dataset is unchanged. See `summary.json` and "
        "`balanced_manifest.csv` for the exact assignment.\n\n"
        "Class 0 has only three independent source groups (one in valid and two in "
        "test), so its estimate has high uncertainty. This re-split also reuses samples "
        "seen in the previous curated valid/test workflow; a final unbiased test requires "
        "newly reviewed, previously unused original groups.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
