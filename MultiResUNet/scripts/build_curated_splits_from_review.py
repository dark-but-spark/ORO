#!/usr/bin/env python
"""Build curated dataset splits from manual review CSV sheets."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_SUFFIXES = [".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--review-csv", required=True, action="append", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument(
        "--selection",
        choices=["eval", "strict-eval", "train-usable"],
        default="strict-eval",
        help=(
            "eval: usable_for_valid=Y and not excluded; "
            "strict-eval: eval plus confidence>=4 and no relabel; "
            "train-usable: usable_for_train=Y and not excluded."
        ),
    )
    parser.add_argument("--min-confidence", type=int, default=4)
    return parser.parse_args()


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def yes(value):
    return (value or "").strip().lower() in {"y", "yes", "1", "true", "是"}


def confidence(value):
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return 0
    return min(parsed, 5)


def find_existing(directory, stem, suffixes):
    for suffix in suffixes:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def copy_or_link(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def selected(row, args):
    if yes(row.get("suggest_exclude")):
        return False, "suggest_exclude"

    if args.selection == "train-usable":
        if yes(row.get("usable_for_train")):
            return True, "usable_for_train"
        return False, "not_usable_for_train"

    if not yes(row.get("usable_for_valid")):
        return False, "not_usable_for_valid"

    if args.selection == "eval":
        return True, "usable_for_valid"

    if yes(row.get("needs_relabel")):
        return False, "needs_relabel"

    if confidence(row.get("label_confidence_1_to_5")) < args.min_confidence:
        return False, "low_confidence"

    return True, "strict_eval"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(output_root):
    text = """train: train/images
val: valid/images
test: test/images

nc: 4
names: ['ballooning', 'fibrosis', 'inflammation', 'steatosis']
"""
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for csv_path in args.review_csv:
        for row in read_rows(csv_path):
            normalized = dict(row)
            normalized["review_csv"] = str(csv_path)
            all_rows.append(normalized)

    manifest_rows = []
    selected_rows = []
    rejected_rows = []
    copy_modes = Counter()
    reject_reasons = Counter()
    split_counts = Counter()
    class_counts = Counter()

    for row in all_rows:
        split = (row.get("split") or "").strip()
        stem = (row.get("image_stem") or Path(row.get("image_file", "")).stem).strip()
        keep, reason = selected(row, args)

        manifest = {
            "review_id": row.get("review_id", ""),
            "split": split,
            "image_stem": stem,
            "classes_present": row.get("classes_present", ""),
            "classes_present_cn": row.get("classes_present_cn", ""),
            "label_confidence_1_to_5": str(confidence(row.get("label_confidence_1_to_5"))),
            "usable_for_valid": row.get("usable_for_valid", ""),
            "usable_for_train": row.get("usable_for_train", ""),
            "needs_relabel": row.get("needs_relabel", ""),
            "suggest_exclude": row.get("suggest_exclude", ""),
            "error_type": row.get("error_type", ""),
            "notes": row.get("notes", ""),
            "curated_keep": str(bool(keep)),
            "decision_reason": reason,
        }

        if not keep:
            reject_reasons[reason] += 1
            rejected_rows.append(manifest)
            manifest_rows.append(manifest)
            continue

        source_split = args.source_root / split
        image_path = find_existing(source_split / "images", stem, IMAGE_SUFFIXES)
        mask_path = find_existing(source_split / "masks", stem, MASK_SUFFIXES)
        label_path = find_existing(source_split / "labels", stem, [".txt"])

        if image_path is None or mask_path is None:
            manifest["curated_keep"] = "False"
            manifest["decision_reason"] = "missing_source_file"
            reject_reasons["missing_source_file"] += 1
            rejected_rows.append(manifest)
            manifest_rows.append(manifest)
            continue

        copy_modes[copy_or_link(image_path, args.output_root / split / "images" / image_path.name, args.copy_mode)] += 1
        copy_modes[copy_or_link(mask_path, args.output_root / split / "masks" / mask_path.name, args.copy_mode)] += 1
        if label_path is not None:
            copy_modes[copy_or_link(label_path, args.output_root / split / "labels" / label_path.name, args.copy_mode)] += 1

        split_counts[split] += 1
        for cls in (row.get("classes_present") or "").split("|"):
            cls = cls.strip()
            if cls:
                class_counts[cls] += 1

        selected_rows.append(manifest)
        manifest_rows.append(manifest)

    fieldnames = [
        "review_id",
        "split",
        "image_stem",
        "classes_present",
        "classes_present_cn",
        "label_confidence_1_to_5",
        "usable_for_valid",
        "usable_for_train",
        "needs_relabel",
        "suggest_exclude",
        "error_type",
        "notes",
        "curated_keep",
        "decision_reason",
    ]
    write_csv(args.output_root / "curated_manifest.csv", manifest_rows, fieldnames)
    write_csv(args.output_root / "selected_rows.csv", selected_rows, fieldnames)
    write_csv(args.output_root / "rejected_rows.csv", rejected_rows, fieldnames)

    write_csv(
        args.output_root / "manual_review_merged.csv",
        all_rows,
        list(all_rows[0].keys()) if all_rows else [],
    )
    write_data_yaml(args.output_root)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "review_csv": [str(path) for path in args.review_csv],
        "selection": args.selection,
        "min_confidence": args.min_confidence,
        "review_rows": len(all_rows),
        "selected_rows": len(selected_rows),
        "rejected_rows": len(rejected_rows),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "reject_reasons": dict(reject_reasons),
        "copy_modes": dict(copy_modes),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Curated Dataset From Manual Review

生成时间：{summary["created_at"]}

## 来源

- source root: `{args.source_root}`
- output root: `{args.output_root}`
- selection: `{args.selection}`
- min confidence: `{args.min_confidence}`

## 选择规则

- `strict-eval`: `usable_for_valid=Y`、`suggest_exclude!=Y`、`needs_relabel!=Y`、`label_confidence_1_to_5 >= min_confidence`
- `eval`: `usable_for_valid=Y`、`suggest_exclude!=Y`
- `train-usable`: `usable_for_train=Y`、`suggest_exclude!=Y`

## 本次统计

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## 输出

- `valid/images`, `valid/masks`, `valid/labels`
- `test/images`, `test/masks`, `test/labels`
- `curated_manifest.csv`: 全部审查行和保留/拒绝原因
- `selected_rows.csv`: 被纳入 curated split 的行
- `rejected_rows.csv`: 被排除的行
- `manual_review_merged.csv`: 原始人工批注合并表
- `data.yaml`: 标准 YOLO/项目类别说明
"""
    (args.output_root / "README_curated_dataset.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
