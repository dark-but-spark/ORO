import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a manual review package from TTA/no-TTA diagnostic CSVs and panels."
    )
    parser.add_argument(
        "--diagnostics-root",
        default="runsTemp/diagnostics",
        help="Root containing best_20260724_tta and best_20260724_no_tta diagnostics.",
    )
    parser.add_argument(
        "--tta-name",
        default="best_20260724_tta",
        help="TTA diagnostics directory name under diagnostics-root.",
    )
    parser.add_argument(
        "--no-tta-name",
        default="best_20260724_no_tta",
        help="No-TTA diagnostics directory name under diagnostics-root.",
    )
    parser.add_argument(
        "--output-dir",
        default="runsTemp/manual_review_package_20260724",
        help="Output package directory.",
    )
    parser.add_argument("--worst-count", type=int, default=50)
    parser.add_argument("--harmful-count", type=int, default=30)
    parser.add_argument("--class2-count", type=int, default=30)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def num(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def build_panel_index(case_dirs):
    index = {}
    for case_dir in case_dirs:
        if not case_dir.exists():
            continue
        for path in case_dir.glob("*.jpg"):
            name = path.name
            # Panel filename format: 001_dice_0.0000_<image_stem>.jpg
            parts = name.split("_", 3)
            if len(parts) < 4:
                continue
            stem = Path(parts[3]).stem
            index.setdefault(stem, path)
    return index


def image_stem(image_file):
    return Path(image_file).stem


def copy_if_available(src, dst_dir, prefix):
    if not src:
        return ""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{prefix}_{src.name}"
    shutil.copy2(src, dst)
    return str(dst)


def classify_auto_hint(row):
    tta = num(row, "tta_overall_dice")
    no_tta = num(row, "no_tta_overall_dice")
    delta = num(row, "tta_minus_no_tta")
    class2_true = num(row, "class_2_true_pixels")
    class2_dice = num(row, "class_2_dice")
    max_prob = max(num(row, f"class_{idx}_prob_mean") for idx in range(4))

    if delta <= -0.20 and no_tta >= 0.20:
        return "D_check_TTA_harm"
    if class2_true > 0 and class2_dice < 0.30:
        return "B_or_C_check_class2"
    if tta < 0.10 and max_prob > 0.02:
        return "A_B_E_check_hard_miss_or_threshold"
    if tta < 0.20:
        return "A_B_check_hard_miss"
    return "review"


def main():
    args = parse_args()
    diagnostics_root = Path(args.diagnostics_root)
    tta_dir = diagnostics_root / args.tta_name
    no_tta_dir = diagnostics_root / args.no_tta_name
    out_dir = Path(args.output_dir)

    tta_csv = tta_dir / "validation_error_report.csv"
    no_tta_csv = no_tta_dir / "validation_error_report.csv"
    if not tta_csv.exists():
        raise FileNotFoundError(f"Missing TTA CSV: {tta_csv}")
    if not no_tta_csv.exists():
        raise FileNotFoundError(f"Missing no-TTA CSV: {no_tta_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tta_rows = read_csv(tta_csv)
    no_tta_rows = read_csv(no_tta_csv)
    no_tta_by_image = {row["image_file"]: row for row in no_tta_rows}

    tta_panel_index = build_panel_index([tta_dir / "worst_cases", tta_dir / "best_cases"])
    no_tta_panel_index = build_panel_index([no_tta_dir / "worst_cases", no_tta_dir / "best_cases"])

    merged = []
    for rank, row in enumerate(tta_rows, start=1):
        no_row = no_tta_by_image.get(row["image_file"], {})
        merged_row = {
            "image_file": row["image_file"],
            "mask_file": row["mask_file"],
            "tta_rank_by_worst": rank,
            "tta_overall_dice": num(row, "overall_dice"),
            "no_tta_overall_dice": num(no_row, "overall_dice"),
            "tta_minus_no_tta": num(row, "overall_dice") - num(no_row, "overall_dice"),
            "tta_overall_jaccard": num(row, "overall_jaccard"),
            "no_tta_overall_jaccard": num(no_row, "overall_jaccard"),
        }
        for idx in range(4):
            for key in ("dice", "jaccard", "fp_pixels", "fn_pixels", "true_pixels", "pred_pixels", "prob_mean"):
                merged_row[f"class_{idx}_{key}"] = num(row, f"class_{idx}_{key}")
        stem = image_stem(row["image_file"])
        merged_row["tta_panel_available"] = "yes" if stem in tta_panel_index else "no"
        merged_row["no_tta_panel_available"] = "yes" if stem in no_tta_panel_index else "no"
        merged_row["auto_hint"] = classify_auto_hint(merged_row)
        merged.append(merged_row)

    tta_worst = merged[: args.worst_count]
    tta_harmful = sorted(merged, key=lambda r: r["tta_minus_no_tta"])[: args.harmful_count]
    class2_hard = sorted(
        [r for r in merged if r["class_2_true_pixels"] > 0],
        key=lambda r: (r["class_2_dice"], r["tta_overall_dice"]),
    )[: args.class2_count]

    selected = []
    seen = set()
    for group_name, rows in (
        ("01_tta_worst", tta_worst),
        ("02_tta_harmful", tta_harmful),
        ("03_class2_hard", class2_hard),
    ):
        group_dir = images_dir / group_name
        for idx, row in enumerate(rows, start=1):
            stem = image_stem(row["image_file"])
            key = (group_name, row["image_file"])
            tta_panel = tta_panel_index.get(stem)
            no_tta_panel = no_tta_panel_index.get(stem)
            row_copy = dict(row)
            row_copy["review_group"] = group_name
            row_copy["review_order"] = idx
            row_copy["tta_panel_package_path"] = copy_if_available(tta_panel, group_dir, f"{idx:03d}_tta")
            row_copy["no_tta_panel_package_path"] = copy_if_available(no_tta_panel, group_dir, f"{idx:03d}_no_tta")
            row_copy["manual_error_type"] = ""
            row_copy["gt_reliable"] = ""
            row_copy["prediction_medically_plausible"] = ""
            row_copy["suggest_fix_label"] = ""
            row_copy["reviewer"] = ""
            row_copy["notes"] = ""
            if key not in seen:
                selected.append(row_copy)
                seen.add(key)

    report_fields = list(merged[0].keys())
    review_fields = [
        "review_group",
        "review_order",
        "image_file",
        "mask_file",
        "tta_overall_dice",
        "no_tta_overall_dice",
        "tta_minus_no_tta",
        "auto_hint",
        "manual_error_type",
        "gt_reliable",
        "prediction_medically_plausible",
        "suggest_fix_label",
        "reviewer",
        "notes",
        "tta_panel_package_path",
        "no_tta_panel_package_path",
    ] + [name for name in report_fields if name not in {
        "image_file", "mask_file", "tta_overall_dice", "no_tta_overall_dice", "tta_minus_no_tta", "auto_hint"
    }]

    write_csv(out_dir / "tta_vs_no_tta_report.csv", merged, report_fields)
    write_csv(out_dir / "manual_review_sheet.csv", selected, review_fields)

    summary = {
        "tta_samples": len(tta_rows),
        "selected_review_rows": len(selected),
        "tta_mean_dice": sum(r["tta_overall_dice"] for r in merged) / len(merged),
        "no_tta_mean_dice": sum(r["no_tta_overall_dice"] for r in merged) / len(merged),
        "mean_tta_delta": sum(r["tta_minus_no_tta"] for r in merged) / len(merged),
        "tta_improved_count": sum(1 for r in merged if r["tta_minus_no_tta"] > 0),
        "tta_worse_count": sum(1 for r in merged if r["tta_minus_no_tta"] < 0),
        "tta_severe_harm_count_delta_lt_minus_0_2": sum(1 for r in merged if r["tta_minus_no_tta"] < -0.2),
        "dice_lt_0_001": sum(1 for r in merged if r["tta_overall_dice"] < 0.001),
        "dice_lt_0_05": sum(1 for r in merged if r["tta_overall_dice"] < 0.05),
        "dice_lt_0_2": sum(1 for r in merged if r["tta_overall_dice"] < 0.2),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

    readme = f"""# Manual Review Package

Generated from:

```text
{tta_dir}
{no_tta_dir}
```

## What Is Inside

```text
manual_review_sheet.csv       Main table for medical/manual review.
tta_vs_no_tta_report.csv      Full machine-generated comparison for all validation images.
summary.json                  Numeric summary.
images/01_tta_worst           TTA worst {args.worst_count} cases.
images/02_tta_harmful         Cases where TTA is much worse than no-TTA.
images/03_class2_hard         Class-2 hard cases.
```

## How To Read Each Panel

Each JPG panel is arranged left to right:

```text
image          original tissue image
ground_truth   human label
prediction     model binary prediction
probability    model probability map
error map      red = false positive, green = false negative
gt_overlay     GT over original image
pred_overlay   prediction over original image
```

## Manual Error Types

Fill `manual_error_type` in `manual_review_sheet.csv` with one main type:

```text
A  Label problem
   GT is clearly wrong, shifted, missing, or labels a wrong structure.

B  Model miss
   GT is reliable, but model misses the target. Probability is also weak.

C  Class confusion
   Model finds a plausible structure, but predicts the wrong class.

D  TTA harm
   no-TTA is acceptable, but TTA becomes clearly worse.

E  Threshold problem
   Probability map is bright at the correct region, but binary prediction disappears.

U  Uncertain
   Reviewer cannot decide confidently.
```

## Suggested Review Procedure

```text
1. Open manual_review_sheet.csv in Excel/WPS.
2. Review images/01_tta_worst first from 001 onward.
3. For each row, compare gt_overlay and pred_overlay.
4. Fill:
   - manual_error_type
   - gt_reliable: yes / no / uncertain
   - prediction_medically_plausible: yes / no / uncertain
   - suggest_fix_label: yes / no
   - notes
5. Then review images/02_tta_harmful to decide whether gated TTA is needed.
6. Finally review images/03_class2_hard to judge whether class2 needs data cleanup or hard mining.
```

## How To Explain This To Medical Students

```text
We are not asking you to evaluate the model score. We need you to judge why the model and the human label disagree.

Please focus on whether the human label is medically reasonable, and whether the model prediction falls on a medically plausible structure.

If the human label is clearly wrong, mark A.
If the human label is correct but the model missed it, mark B.
If the model detects a plausible structure but the class is wrong, mark C.
If no-TTA is good but TTA is bad, mark D.
If the probability map sees the right place but the final mask disappears, mark E.
Use U if uncertain.
```

## Machine Summary

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```
"""
    (out_dir / "README_manual_review.md").write_text(readme, encoding="utf-8")
    print(f"Manual review package created: {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
