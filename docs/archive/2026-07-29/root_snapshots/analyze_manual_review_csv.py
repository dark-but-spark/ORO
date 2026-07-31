import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def clean(value):
    return (value or "").strip()


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_csv(path):
    last_error = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f)), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def counter_table(rows, column):
    counts = Counter(clean(r.get(column)) or "<blank>" for r in rows)
    return counts.most_common()


def grouped_counter(rows, group_col, value_col):
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[clean(row.get(group_col)) or "<blank>"][clean(row.get(value_col)) or "<blank>"] += 1
    return grouped


def write_count_csv(path, title, counts):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "value", "count"])
        for value, count in counts:
            writer.writerow([title, value, count])


def main():
    parser = argparse.ArgumentParser(description="Analyze manual review CSV results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("runsTemp/manual_review_package_20260724/副本manual_review_sheet.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runsTemp/manual_review_package_20260724/manual_review_analysis_20260727"),
    )
    args = parser.parse_args()

    rows, encoding = read_csv(args.csv)
    filled = [
        r for r in rows
        if clean(r.get("manual_error_type"))
        or clean(r.get("gt_reliable"))
        or clean(r.get("prediction_medically_plausible"))
        or clean(r.get("suggest_fix_label"))
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sections = {}
    for col in (
        "review_group",
        "manual_error_type",
        "gt_reliable",
        "prediction_medically_plausible",
        "suggest_fix_label",
        "auto_hint",
    ):
        sections[col] = counter_table(filled, col)
        write_count_csv(args.output_dir / f"{col}_counts.csv", col, sections[col])

    by_group_error = grouped_counter(filled, "review_group", "manual_error_type")
    by_group_gt = grouped_counter(filled, "review_group", "gt_reliable")
    by_gt_fix = grouped_counter(filled, "gt_reliable", "suggest_fix_label")

    metric_rows = []
    for error_type in sorted({clean(r.get("manual_error_type")) for r in filled if clean(r.get("manual_error_type"))}):
        subset = [r for r in filled if clean(r.get("manual_error_type")) == error_type]
        metric_rows.append({
            "manual_error_type": error_type,
            "count": len(subset),
            "tta_mean_dice": mean(number(r.get("tta_overall_dice")) for r in subset),
            "no_tta_mean_dice": mean(number(r.get("no_tta_overall_dice")) for r in subset),
            "tta_delta_mean": mean(number(r.get("tta_minus_no_tta")) for r in subset),
            "class2_true_pixels_mean": mean(number(r.get("class_2_true_pixels")) for r in subset),
            "class2_pred_pixels_mean": mean(number(r.get("class_2_pred_pixels")) for r in subset),
            "class2_dice_mean": mean(number(r.get("class_2_dice")) for r in subset),
        })

    with (args.output_dir / "metric_by_manual_error_type.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(metric_rows[0].keys()) if metric_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(metric_rows)

    deltas = [number(r.get("tta_minus_no_tta")) for r in filled]
    deltas = [d for d in deltas if not math.isnan(d)]
    class2_hard = [r for r in filled if clean(r.get("review_group")) == "03_class2_hard"]

    md = []
    md.append("# Manual Review Analysis")
    md.append("")
    md.append(f"Source CSV: `{args.csv}`")
    md.append(f"Encoding used: `{encoding}`")
    md.append("")
    md.append("## Coverage")
    md.append("")
    md.append(f"- Total rows: {len(rows)}")
    md.append(f"- Filled rows: {len(filled)}")
    md.append(f"- Unique filled images: {len(set(r.get('image_file') for r in filled))}")
    md.append("")

    for col, counts in sections.items():
        md.append(f"## {col}")
        md.append("")
        md.append("| Value | Count |")
        md.append("| --- | ---: |")
        for value, count in counts:
            md.append(f"| {value} | {count} |")
        md.append("")

    md.append("## Manual Error Type By Group")
    md.append("")
    for group in sorted(by_group_error):
        md.append(f"- {group}: {dict(by_group_error[group])}")
    md.append("")

    md.append("## GT Reliability By Group")
    md.append("")
    for group in sorted(by_group_gt):
        md.append(f"- {group}: {dict(by_group_gt[group])}")
    md.append("")

    md.append("## TTA Delta")
    md.append("")
    md.append(f"- Mean delta: {fmt(mean(deltas))}")
    md.append(f"- Median delta: {fmt(statistics.median(deltas) if deltas else float('nan'))}")
    md.append(f"- TTA worse count: {sum(d < 0 for d in deltas)}")
    md.append(f"- TTA delta < -0.05: {sum(d < -0.05 for d in deltas)}")
    md.append(f"- TTA delta < -0.20: {sum(d < -0.20 for d in deltas)}")
    md.append("")

    md.append("## Class2 Hard")
    md.append("")
    md.append(f"- Filled class2 hard rows: {len(class2_hard)}")
    if class2_hard:
        md.append(f"- Error types: {dict(Counter(clean(r.get('manual_error_type')) or '<blank>' for r in class2_hard))}")
        md.append(f"- GT reliable: {dict(Counter(clean(r.get('gt_reliable')) or '<blank>' for r in class2_hard))}")
        md.append(f"- Suggest fix label: {dict(Counter(clean(r.get('suggest_fix_label')) or '<blank>' for r in class2_hard))}")
        md.append(f"- Mean class2 Dice: {fmt(mean(number(r.get('class_2_dice')) for r in class2_hard))}")
        md.append(f"- Mean class2 true pixels: {fmt(mean(number(r.get('class_2_true_pixels')) for r in class2_hard))}")
        md.append(f"- Mean class2 pred pixels: {fmt(mean(number(r.get('class_2_pred_pixels')) for r in class2_hard))}")
    md.append("")

    md.append("## Metrics By Manual Error Type")
    md.append("")
    md.append("| Error | Count | TTA Dice | no-TTA Dice | TTA Delta | class2 Dice | class2 true px | class2 pred px |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in metric_rows:
        md.append(
            f"| {r['manual_error_type']} | {r['count']} | {fmt(r['tta_mean_dice'])} | "
            f"{fmt(r['no_tta_mean_dice'])} | {fmt(r['tta_delta_mean'])} | "
            f"{fmt(r['class2_dice_mean'])} | {fmt(r['class2_true_pixels_mean'])} | "
            f"{fmt(r['class2_pred_pixels_mean'])} |"
        )
    md.append("")

    report_path = args.output_dir / "manual_review_analysis.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Read {len(rows)} rows, filled {len(filled)} rows, encoding={encoding}")
    print(f"Report: {report_path}")
    for col in ("review_group", "manual_error_type", "gt_reliable", "prediction_medically_plausible", "suggest_fix_label"):
        print(col, dict(sections[col]))
    print("TTA worse:", sum(d < 0 for d in deltas), "of", len(deltas))


if __name__ == "__main__":
    main()
