import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


DEFAULT_DIAGNOSTICS_DIR = Path("runsTemp") / "diagnostics"


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    return f"{value:.6f}"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def float_field(row, name, default=0.0):
    value = row.get(name, "")
    if value == "":
        return default
    return float(value)


def class_indices_from_summary(summary):
    classes = []
    for key in summary.get("per_class", {}):
        if key.startswith("class_"):
            try:
                classes.append(int(key.split("_", 1)[1]))
            except ValueError:
                pass
    return sorted(classes)


def dice_from_counts(tp, fp, fn, eps=1e-7):
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def jaccard_from_counts(tp, fp, fn, eps=1e-7):
    return (tp + eps) / (tp + fp + fn + eps)


def summarize_run(run_dir, ignore_classes):
    summary_path = run_dir / "per_class_summary.json"
    report_path = run_dir / "validation_error_report.csv"
    if not summary_path.exists() or not report_path.exists():
        return None, []

    summary = load_json(summary_path)
    rows = read_rows(report_path)
    all_classes = class_indices_from_summary(summary)
    kept_classes = [idx for idx in all_classes if idx not in ignore_classes]

    per_class = summary.get("per_class", {})
    macro_all_dice = mean(per_class[f"class_{idx}"]["mean_dice"] for idx in all_classes)
    macro_kept_dice = mean(per_class[f"class_{idx}"]["mean_dice"] for idx in kept_classes)
    macro_all_jaccard = mean(per_class[f"class_{idx}"]["mean_jaccard"] for idx in all_classes)
    macro_kept_jaccard = mean(per_class[f"class_{idx}"]["mean_jaccard"] for idx in kept_classes)

    sample_rows = []
    kept_macro_dice_values = []
    kept_macro_jaccard_values = []
    kept_tp = 0.0
    kept_fp = 0.0
    kept_fn = 0.0
    all_tp = 0.0
    all_fp = 0.0
    all_fn = 0.0

    for row in rows:
        kept_dice = []
        kept_jaccard = []
        for idx in all_classes:
            true_pixels = float_field(row, f"class_{idx}_true_pixels")
            pred_pixels = float_field(row, f"class_{idx}_pred_pixels")
            fp_pixels = float_field(row, f"class_{idx}_fp_pixels")
            fn_pixels = float_field(row, f"class_{idx}_fn_pixels")
            tp_pixels = max(0.0, true_pixels - fn_pixels)

            all_tp += tp_pixels
            all_fp += fp_pixels
            all_fn += fn_pixels

            if idx in kept_classes:
                kept_tp += tp_pixels
                kept_fp += fp_pixels
                kept_fn += fn_pixels
                kept_dice.append(float_field(row, f"class_{idx}_dice"))
                kept_jaccard.append(float_field(row, f"class_{idx}_jaccard"))

        sample_macro_dice = mean(kept_dice)
        sample_macro_jaccard = mean(kept_jaccard)
        kept_macro_dice_values.append(sample_macro_dice)
        kept_macro_jaccard_values.append(sample_macro_jaccard)
        sample_rows.append(
            {
                "sample_index": row.get("sample_index", ""),
                "image_file": row.get("image_file", ""),
                "original_overall_dice": float_field(row, "overall_dice"),
                "original_macro_class_dice": float_field(row, "macro_class_dice"),
                "ignore_class_macro_dice": sample_macro_dice,
                "original_overall_jaccard": float_field(row, "overall_jaccard"),
                "original_macro_class_jaccard": float_field(row, "macro_class_jaccard"),
                "ignore_class_macro_jaccard": sample_macro_jaccard,
            }
        )

    sample_rows.sort(key=lambda item: item["ignore_class_macro_dice"])

    result = {
        "run_name": run_dir.name,
        "validation_samples": int(summary.get("validation_samples", len(rows))),
        "val_tta": summary.get("val_tta", "unknown"),
        "threshold": summary.get("threshold"),
        "ignored_classes": ",".join(str(idx) for idx in sorted(ignore_classes)),
        "kept_classes": ",".join(str(idx) for idx in kept_classes),
        "summary_mean_overall_dice": float(summary.get("mean_overall_dice", float("nan"))),
        "summary_mean_overall_jaccard": float(summary.get("mean_overall_jaccard", float("nan"))),
        "summary_macro_all_class_dice": macro_all_dice,
        "summary_macro_without_ignored_dice": macro_kept_dice,
        "summary_macro_all_class_jaccard": macro_all_jaccard,
        "summary_macro_without_ignored_jaccard": macro_kept_jaccard,
        "sample_mean_macro_without_ignored_dice": mean(kept_macro_dice_values),
        "sample_median_macro_without_ignored_dice": median(kept_macro_dice_values),
        "sample_min_macro_without_ignored_dice": min(kept_macro_dice_values),
        "sample_mean_macro_without_ignored_jaccard": mean(kept_macro_jaccard_values),
        "micro_all_class_dice": dice_from_counts(all_tp, all_fp, all_fn),
        "micro_without_ignored_dice": dice_from_counts(kept_tp, kept_fp, kept_fn),
        "micro_all_class_jaccard": jaccard_from_counts(all_tp, all_fp, all_fn),
        "micro_without_ignored_jaccard": jaccard_from_counts(kept_tp, kept_fp, kept_fn),
    }
    return result, sample_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summaries, ignore_classes):
    lines = [
        "# Ignore Class Analysis",
        "",
        f"Ignored classes: {', '.join(str(idx) for idx in sorted(ignore_classes))}",
        "",
        "## Summary",
        "",
        "| Run | TTA | Original overall Dice | Macro Dice all classes | Macro Dice without ignored | Micro Dice all classes | Micro Dice without ignored |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {run_name} | {val_tta} | {overall} | {macro_all} | {macro_kept} | {micro_all} | {micro_kept} |".format(
                run_name=row["run_name"],
                val_tta=row["val_tta"],
                overall=fmt(row["summary_mean_overall_dice"]),
                macro_all=fmt(row["summary_macro_all_class_dice"]),
                macro_kept=fmt(row["summary_macro_without_ignored_dice"]),
                micro_all=fmt(row["micro_all_class_dice"]),
                micro_kept=fmt(row["micro_without_ignored_dice"]),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Original overall Dice` comes from the diagnostic summary and keeps the original evaluation behavior.",
            "- `Macro Dice without ignored` averages the remaining class Dice values, so empty/easy classes can make it much higher than original overall Dice.",
            "- `Micro Dice without ignored` sums TP/FP/FN over the remaining classes before calculating Dice. This is usually the stricter and more useful estimate.",
            "- This script only ignores the class during scoring. It does not remove slices, change labels, or retrain the model.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute diagnostic scores while ignoring selected classes, e.g. ignore class2."
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_DIR,
        help="Directory containing diagnostic run folders with per_class_summary.json and validation_error_report.csv.",
    )
    parser.add_argument(
        "--ignore-classes",
        type=int,
        nargs="+",
        default=[2],
        help="Class indices to ignore during score recomputation. Default: 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runsTemp") / "diagnostics_ignore_class_analysis",
        help="Directory for generated CSV and Markdown reports.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ignore_classes = set(args.ignore_classes)
    summaries = []

    for run_dir in sorted(p for p in args.diagnostics_dir.iterdir() if p.is_dir()):
        summary, sample_rows = summarize_run(run_dir, ignore_classes)
        if summary is None:
            continue
        summaries.append(summary)
        write_csv(
            args.output_dir / f"{run_dir.name}_ignore_classes_{'_'.join(map(str, sorted(ignore_classes)))}_samples.csv",
            sample_rows,
            [
                "sample_index",
                "image_file",
                "original_overall_dice",
                "original_macro_class_dice",
                "ignore_class_macro_dice",
                "original_overall_jaccard",
                "original_macro_class_jaccard",
                "ignore_class_macro_jaccard",
            ],
        )

    summaries.sort(key=lambda item: item["micro_without_ignored_dice"], reverse=True)
    summary_fields = [
        "run_name",
        "validation_samples",
        "val_tta",
        "threshold",
        "ignored_classes",
        "kept_classes",
        "summary_mean_overall_dice",
        "summary_mean_overall_jaccard",
        "summary_macro_all_class_dice",
        "summary_macro_without_ignored_dice",
        "summary_macro_all_class_jaccard",
        "summary_macro_without_ignored_jaccard",
        "sample_mean_macro_without_ignored_dice",
        "sample_median_macro_without_ignored_dice",
        "sample_min_macro_without_ignored_dice",
        "sample_mean_macro_without_ignored_jaccard",
        "micro_all_class_dice",
        "micro_without_ignored_dice",
        "micro_all_class_jaccard",
        "micro_without_ignored_jaccard",
    ]
    write_csv(args.output_dir / "ignore_class_summary.csv", summaries, summary_fields)
    write_markdown(args.output_dir / "ignore_class_report.md", summaries, ignore_classes)

    print("Ignore-class analysis complete.")
    print(f"Diagnostics dir: {args.diagnostics_dir}")
    print(f"Output dir: {args.output_dir}")
    print()
    for row in summaries:
        print(
            f"{row['run_name']}: original_overall={fmt(row['summary_mean_overall_dice'])}, "
            f"macro_without_class={fmt(row['summary_macro_without_ignored_dice'])}, "
            f"micro_without_class={fmt(row['micro_without_ignored_dice'])}"
        )


if __name__ == "__main__":
    main()
