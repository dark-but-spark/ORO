import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "valid", "test")


def original_stem(path):
    stem = Path(path).stem
    marker = ".rf."
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def image_files(img_dir):
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return []
    return sorted(
        path for path in img_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_groups(source_root, splits=DEFAULT_SPLITS):
    source_root = Path(source_root)
    groups = defaultdict(list)
    split_counts = {}

    for split in splits:
        img_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"
        files = image_files(img_dir)
        split_counts[split] = len(files)

        for img_path in files:
            label_path = label_dir / f"{img_path.stem}.txt"
            groups[original_stem(img_path)].append(
                {
                    "source_split": split,
                    "image": str(img_path),
                    "label": str(label_path) if label_path.exists() else None,
                    "stem": img_path.stem,
                }
            )

    return dict(groups), split_counts


def leakage_report(groups):
    split_sets = {split: set() for split in DEFAULT_SPLITS}
    for group_id, records in groups.items():
        for record in records:
            split_sets[record["source_split"]].add(group_id)

    pair_rows = []
    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for left, right in pairs:
        overlap = sorted(split_sets[left] & split_sets[right])
        pair_rows.append(
            {
                "pair": f"{left}-{right}",
                "overlap_originals": len(overlap),
                "left_originals": len(split_sets[left]),
                "right_originals": len(split_sets[right]),
                "pct_of_left": len(overlap) / max(1, len(split_sets[left])),
                "pct_of_right": len(overlap) / max(1, len(split_sets[right])),
                "examples": overlap[:20],
            }
        )

    return {
        "split_original_counts": {split: len(ids) for split, ids in split_sets.items()},
        "overlap": pair_rows,
    }


def split_groups(groups, train_ratio, val_ratio, test_ratio, seed):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    n_total = len(group_ids)
    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)
    n_train = min(n_train, n_total)
    n_val = min(n_val, n_total - n_train)

    assignments = {}
    for group_id in group_ids[:n_train]:
        assignments[group_id] = "train"
    for group_id in group_ids[n_train:n_train + n_val]:
        assignments[group_id] = "valid"
    for group_id in group_ids[n_train + n_val:]:
        assignments[group_id] = "test"

    return assignments


def selected_records(records, max_variants_per_group, seed, group_id):
    records = sorted(records, key=lambda record: (record["source_split"], record["stem"]))
    if max_variants_per_group is None or len(records) <= max_variants_per_group:
        return records

    rng = random.Random(f"{seed}:{group_id}")
    shuffled = records[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:max_variants_per_group], key=lambda record: (record["source_split"], record["stem"]))


def transfer_file(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def write_data_yaml(output_root, source_yaml):
    output_root = Path(output_root)
    if source_yaml and Path(source_yaml).exists():
        text = Path(source_yaml).read_text(encoding="utf-8")
        (output_root / "data.source.yaml").write_text(text, encoding="utf-8")

    yaml_text = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 4\n"
        "names: ['ballooning', 'fibrosis', 'inflammation', 'steatosis']\n"
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def create_clean_split(args):
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    groups, source_split_counts = collect_groups(source_root)
    report_before = leakage_report(groups)
    assignments = split_groups(
        groups,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "valid": args.val_ratio,
            "test": args.test_ratio,
        },
        "max_variants_per_group": args.max_variants_per_group,
        "copy_mode": args.copy_mode,
        "source_split_image_counts": source_split_counts,
        "source_leakage_report": report_before,
        "groups": {},
        "output_counts": {
            "train": {"groups": 0, "images": 0, "missing_labels": 0},
            "valid": {"groups": 0, "images": 0, "missing_labels": 0},
            "test": {"groups": 0, "images": 0, "missing_labels": 0},
        },
    }

    for split in DEFAULT_SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    for group_id in tqdm(sorted(groups), desc="Creating group-safe split"):
        split = assignments[group_id]
        records = selected_records(groups[group_id], args.max_variants_per_group, args.seed, group_id)
        manifest["output_counts"][split]["groups"] += 1
        manifest["groups"][group_id] = {
            "assigned_split": split,
            "source_splits": sorted({record["source_split"] for record in groups[group_id]}),
            "all_variants": len(groups[group_id]),
            "selected_variants": len(records),
            "files": [],
        }

        for record in records:
            img_src = Path(record["image"])
            img_dst = output_root / split / "images" / img_src.name
            transfer_file(img_src, img_dst, args.copy_mode)

            label_src = Path(record["label"]) if record["label"] else None
            label_dst = output_root / split / "labels" / f"{img_src.stem}.txt"
            if label_src and label_src.exists():
                transfer_file(label_src, label_dst, args.copy_mode)
            else:
                label_dst.write_text("", encoding="utf-8")
                manifest["output_counts"][split]["missing_labels"] += 1

            manifest["output_counts"][split]["images"] += 1
            manifest["groups"][group_id]["files"].append(
                {
                    "source_split": record["source_split"],
                    "image": img_src.name,
                    "label": label_dst.name,
                }
            )

    write_data_yaml(output_root, source_root / "data.yaml")

    report_after, _ = collect_groups(output_root)
    manifest["output_leakage_report"] = leakage_report(report_after)

    manifest_path = output_root / "group_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.convert_masks:
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from yolo_to_npz import convert_standard_splits

        convert_standard_splits(output_root, DEFAULT_SPLITS, num_classes=args.num_classes, overwrite=True)

    print(json.dumps({
        "output_root": str(output_root),
        "output_counts": manifest["output_counts"],
        "source_leakage_overlap": manifest["source_leakage_report"]["overlap"],
        "output_leakage_overlap": manifest["output_leakage_report"]["overlap"],
        "manifest": str(manifest_path),
    }, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe train/valid/test split for Roboflow-exported YOLO datasets."
    )
    parser.add_argument("--source-root", required=True, help="Original YOLO dataset root with train/valid/test folders.")
    parser.add_argument("--output-root", required=True, help="Clean dataset output root.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-variants-per-group", type=int, default=None,
                        help="Optional cap on augmented variants kept per original image group.")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy",
                        help="How files are placed in the clean dataset. hardlink saves space on same filesystem.")
    parser.add_argument("--convert-masks", action="store_true",
                        help="Convert copied YOLO labels to .npz masks after splitting.")
    parser.add_argument("--num-classes", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    create_clean_split(parse_args())
