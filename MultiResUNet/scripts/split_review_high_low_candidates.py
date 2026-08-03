#!/usr/bin/env python
"""Split review sheets into high-quality and low-quality candidate packages."""

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--high-per-split", type=int, default=120)
    parser.add_argument("--low-per-split", type=int, default=80)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(value, default=0):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def quality_key(row):
    present_count = len([item for item in (row.get("classes_present") or "").split("|") if item])
    return (
        to_int(row.get("auto_priority_score")),
        to_int(row.get("component_total")),
        to_int(row.get("small_component_count")),
        present_count,
        abs(to_float(row.get("largest_component_ratio_min")) - 1.0),
        row.get("review_id", ""),
    )


def problem_key(row):
    present_count = len([item for item in (row.get("classes_present") or "").split("|") if item])
    return (
        -to_int(row.get("auto_priority_score")),
        -to_int(row.get("component_total")),
        -to_int(row.get("small_component_count")),
        -present_count,
        row.get("review_id", ""),
    )


def copy_panel(review_root, output_dir, group, row, order):
    rel = row.get("review_image", "")
    if not rel:
        return ""
    src = review_root / rel
    if not src.exists():
        return ""
    split = row.get("split", "unknown")
    dst_dir = output_dir / "images" / group / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix or ".jpg"
    dst = dst_dir / f"{order:04d}_{row.get('review_id', 'sample')}{suffix}"
    shutil.copy2(src, dst)
    return str(dst.relative_to(output_dir)).replace("\\", "/")


def annotate_rows(review_root, output_dir, rows, group):
    annotated = []
    for order, row in enumerate(rows, start=1):
        copied = dict(row)
        copied["candidate_group"] = group
        copied["candidate_order"] = str(order)
        copied["candidate_review_image"] = copy_panel(review_root, output_dir, group, row, order)
        copied["manual_priority"] = copied.get("manual_priority", "")
        copied["label_confidence_1_to_5"] = copied.get("label_confidence_1_to_5", "")
        copied["usable_for_valid"] = copied.get("usable_for_valid", "")
        copied["needs_relabel"] = copied.get("needs_relabel", "")
        copied["suggest_exclude"] = copied.get("suggest_exclude", "")
        copied["reviewer"] = copied.get("reviewer", "")
        copied["notes"] = copied.get("notes", "")
        annotated.append(copied)
    return annotated


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    high_all = []
    low_all = []
    summary = {}

    for split in args.splits:
        sheet = args.review_root / f"manual_review_sheet_{args.dataset}_{split}.csv"
        rows = read_csv(sheet)
        high_rows = sorted(rows, key=quality_key)[: args.high_per_split]
        low_rows = sorted(rows, key=problem_key)[: args.low_per_split]

        high_annotated = annotate_rows(args.review_root, args.output_dir, high_rows, "high_quality_candidates")
        low_annotated = annotate_rows(args.review_root, args.output_dir, low_rows, "low_quality_problem_candidates")
        high_all.extend(high_annotated)
        low_all.extend(low_annotated)

        summary[split] = {
            "source_rows": len(rows),
            "high_rows": len(high_rows),
            "low_rows": len(low_rows),
            "high_score_range": [
                min(to_int(r.get("auto_priority_score")) for r in high_rows) if high_rows else None,
                max(to_int(r.get("auto_priority_score")) for r in high_rows) if high_rows else None,
            ],
            "low_score_range": [
                min(to_int(r.get("auto_priority_score")) for r in low_rows) if low_rows else None,
                max(to_int(r.get("auto_priority_score")) for r in low_rows) if low_rows else None,
            ],
        }

    fieldnames = ["candidate_group", "candidate_order", "candidate_review_image"]
    base_fields = list(high_all[0].keys()) if high_all else list(low_all[0].keys())
    fieldnames.extend([name for name in base_fields if name not in fieldnames])

    write_csv(args.output_dir / "high_quality_candidates.csv", high_all, fieldnames)
    write_csv(args.output_dir / "low_quality_problem_candidates.csv", low_all, fieldnames)
    write_csv(args.output_dir / "manual_review_sheet_high_low.csv", high_all + low_all, fieldnames)

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "review_root": str(args.review_root),
        "dataset": args.dataset,
        "output_dir": str(args.output_dir),
        "high_per_split": args.high_per_split,
        "low_per_split": args.low_per_split,
        "summary": summary,
        "high_classes": dict(Counter(row.get("classes_present_cn", "") for row in high_all)),
        "low_classes": dict(Counter(row.get("classes_present_cn", "") for row in low_all)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = f"""# A valid/test 高低分候选审查包

生成时间：{summary_payload["created_at"]}

## 定义

这里的“高分数”不是模型 Dice，而是机器综合质量评分后的高可信候选：

- 高分数/高可信集合：`auto_priority_score` 低、连通区域少、小碎片少、类别结构简单。
- 低分数/问题集合：`auto_priority_score` 高、标注面积异常、连通区域多、小碎片多或类别复杂。

## 文件

- `high_quality_candidates.csv`：建议第一轮人工筛，目标是建立干净 valid/test。
- `low_quality_problem_candidates.csv`：建议第二轮人工筛，目标是找标注问题、剔除或重标。
- `manual_review_sheet_high_low.csv`：两组合并表。
- `images/high_quality_candidates/`：高可信候选图。
- `images/low_quality_problem_candidates/`：问题候选图。

## 本次数量

```json
{json.dumps(summary_payload, ensure_ascii=False, indent=2)}
```

## 人工填写建议

高可信集合重点填：

- `label_confidence_1_to_5`
- `usable_for_valid`
- `needs_relabel`
- `suggest_exclude`
- `notes`

低分数集合重点填：

- `error_type`
- `needs_relabel`
- `suggest_exclude`
- `usable_for_train`
- `notes`
"""
    (args.output_dir / "README_high_low_review.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
