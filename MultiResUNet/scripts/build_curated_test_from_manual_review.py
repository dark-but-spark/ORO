#!/usr/bin/env python
"""Build a curated test subset from manual worst-case review records."""

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


LABEL_ACTIONS = {
    "fix_label",
    "fix lable",
    "gt漏标",
    "gt错标",
    "gt漏标错标",
}
EXCLUDE_ACTIONS = {
    "exclude_eval",
    "exclude eral",
}
MODEL_ERROR_ACTIONS = {
    "keep_model_error",
    "keep model error",
    "模型漏检，改善模型",
    "改善模型",
    "类别混淆，改善模型",
}
THRESHOLD_ACTIONS = {"threshold check"}


def normalize_text(value):
    return (value or "").strip()


def normalize_action(value):
    action = normalize_text(value).lower()
    if action in LABEL_ACTIONS:
        return "label_problem"
    if action in EXCLUDE_ACTIONS:
        return "exclude_eval"
    if action in MODEL_ERROR_ACTIONS:
        return "model_error"
    if action in THRESHOLD_ACTIONS:
        return "threshold_check"
    return "review_note"


def read_review_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_images(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


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


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_review_index(rows):
    by_image = defaultdict(list)
    normalized_rows = []
    for row in rows:
        image_file = normalize_text(row.get("image_file"))
        action_norm = normalize_action(row.get("suggest_action"))
        gt_reliable = normalize_text(row.get("gt_reliable")).upper()
        pred_plausible = normalize_text(row.get("prediction_medically_plausible")).upper()
        normalized = dict(row)
        normalized["normalized_action"] = action_norm
        normalized["gt_reliable_norm"] = gt_reliable
        normalized["prediction_medically_plausible_norm"] = pred_plausible
        normalized_rows.append(normalized)
        if image_file:
            by_image[image_file].append(normalized)
    return normalized_rows, by_image


def classify_image(image_file, rows):
    if not rows:
        return {
            "curated_keep": True,
            "status": "unreviewed_not_worst",
            "reason": "not_present_in_manual_worst_case_review",
        }

    actions = {row["normalized_action"] for row in rows}
    gt_values = {row["gt_reliable_norm"] for row in rows}
    if "exclude_eval" in actions:
        return {"curated_keep": False, "status": "excluded", "reason": "manual_exclude_eval"}
    if "label_problem" in actions:
        return {"curated_keep": False, "status": "label_problem", "reason": "manual_label_fix_needed"}
    if "N" in gt_values:
        return {"curated_keep": False, "status": "gt_unreliable", "reason": "gt_reliable_N"}
    if actions <= {"model_error", "threshold_check", "review_note"}:
        return {"curated_keep": True, "status": "reviewed_trusted", "reason": "gt_reliable_and_no_label_exclusion"}
    return {"curated_keep": True, "status": "reviewed_trusted", "reason": "no_exclusion_rule_triggered"}


def make_readme(args, summary):
    return f"""# Curated Test Dataset From Manual Review

生成时间：{summary["created_at"]}

## 来源

- 原始 test：`{args.source_test_root}`
- 人工审查表：`{args.review_csv}`
- 输出目录：`{args.output_root}`

## 判定规则

这是一个相对保守的 curated test。对出现在人工 worst-case 审查表中的图片：

- 任一记录 `suggest_action` 属于 `exclude_eval`，排除。
- 任一记录 `suggest_action` 属于 `fix_label / GT漏标 / GT错标 / GT漏标错标`，排除。
- 任一记录 `gt_reliable = N`，排除。
- 其余人工确认 GT 相对可靠的样本保留。
- 没出现在 worst-case 审查表中的原始 test 样本暂时保留，状态记为 `unreviewed_not_worst`。

## 数量

- 原始 test 图片数：{summary["source_images"]}
- 人工审查记录数：{summary["review_rows"]}
- 人工审查涉及去重图片数：{summary["reviewed_unique_images"]}
- curated 保留图片数：{summary["kept_images"]}
- 排除图片数：{summary["excluded_images"]}

## 输出文件

- `test/images/`：curated test 图片。
- `test/masks/`：curated test NPZ mask。
- `test/labels/`：如果原始 YOLO txt 存在，同步保留。
- `curated_manifest.csv`：每张原始 test 图片的保留/排除状态。
- `excluded_images.csv`：被排除的图片及原因。
- `label_problem_images.csv`：需要优先修正 GT 的图片。
- `reviewed_trusted_images.csv`：人工审查后仍可保留的 worst-case 图片。
- `manual_review_normalized.csv`：带归一化字段的人工审查表。
- `summary.json`：机器可读统计。

## 使用建议

训练仍然使用原始固定 train/valid。最终报告建议同时给出：

1. 原始 test 分数。
2. curated test 分数。
3. curated test 忽略 class2 的分数。

curated test 不应该反向参与训练或调参，只用于更干净地解释最终泛化性能。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--source-test-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()

    image_dir = args.source_test_root / "images"
    mask_dir = args.source_test_root / "masks"
    label_dir = args.source_test_root / "labels"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/masks under {args.source_test_root}")

    rows = read_review_rows(args.review_csv)
    normalized_rows, review_by_image = build_review_index(rows)
    images = list_images(image_dir)

    manifest_rows = []
    excluded_rows = []
    label_problem_rows = []
    reviewed_trusted_rows = []
    copy_modes = Counter()
    status_counts = Counter()

    for image_path in images:
        image_file = image_path.name
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}.npz"
        label_path = label_dir / f"{stem}.txt"
        review_rows = review_by_image.get(image_file, [])
        decision = classify_image(image_file, review_rows)
        status_counts[decision["status"]] += 1

        row = {
            "image_file": image_file,
            "mask_file": mask_path.name,
            "curated_keep": str(decision["curated_keep"]),
            "status": decision["status"],
            "reason": decision["reason"],
            "review_records": len(review_rows),
            "review_actions": "|".join(sorted({r["normalized_action"] for r in review_rows})),
            "gt_reliable_values": "|".join(sorted({r["gt_reliable_norm"] for r in review_rows})),
            "prediction_plausible_values": "|".join(sorted({r["prediction_medically_plausible_norm"] for r in review_rows})),
        }
        manifest_rows.append(row)

        if decision["curated_keep"]:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for kept image: {mask_path}")
            copy_modes[copy_or_link(image_path, args.output_root / "test" / "images" / image_path.name, args.copy_mode)] += 1
            copy_modes[copy_or_link(mask_path, args.output_root / "test" / "masks" / mask_path.name, args.copy_mode)] += 1
            if label_path.exists():
                copy_modes[copy_or_link(label_path, args.output_root / "test" / "labels" / label_path.name, args.copy_mode)] += 1
            if review_rows:
                reviewed_trusted_rows.append(row)
        else:
            excluded_rows.append(row)
            if decision["status"] == "label_problem":
                label_problem_rows.append(row)

    review_fields = list(normalized_rows[0].keys()) if normalized_rows else []
    manifest_fields = [
        "image_file",
        "mask_file",
        "curated_keep",
        "status",
        "reason",
        "review_records",
        "review_actions",
        "gt_reliable_values",
        "prediction_plausible_values",
    ]
    write_csv(args.output_root / "manual_review_normalized.csv", normalized_rows, review_fields)
    write_csv(args.output_root / "curated_manifest.csv", manifest_rows, manifest_fields)
    write_csv(args.output_root / "excluded_images.csv", excluded_rows, manifest_fields)
    write_csv(args.output_root / "label_problem_images.csv", label_problem_rows, manifest_fields)
    write_csv(args.output_root / "reviewed_trusted_images.csv", reviewed_trusted_rows, manifest_fields)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "review_csv": str(args.review_csv),
        "source_test_root": str(args.source_test_root),
        "output_root": str(args.output_root),
        "source_images": len(images),
        "review_rows": len(rows),
        "reviewed_unique_images": len(review_by_image),
        "kept_images": sum(1 for row in manifest_rows if row["curated_keep"] == "True"),
        "excluded_images": len(excluded_rows),
        "status_counts": dict(status_counts),
        "copy_modes": dict(copy_modes),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "README.md").write_text(make_readme(args, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
