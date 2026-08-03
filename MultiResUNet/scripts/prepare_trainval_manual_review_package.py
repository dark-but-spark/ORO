import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = {".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CLASS_NAMES = ["ballooning", "fibrosis", "inflammation", "steatosis"]
CLASS_NAMES_CN = ["气球样变", "纤维化", "炎症", "脂肪变"]

# BGR colors for OpenCV output.
CLASS_COLORS = [
    (0, 0, 255),      # class0 red
    (0, 180, 0),      # class1 green
    (255, 0, 0),      # class2 blue
    (0, 220, 255),    # class3 yellow
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create manual-review panels and CSV sheets from image/mask datasets."
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=["data/20260204111923", "data/385-liver.groupclean.v1"],
        help="Dataset roots containing split folders such as train/valid/test.",
    )
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument(
        "--priority-strategy",
        choices=["comprehensive", "class2"],
        default="comprehensive",
        help="Priority rule. comprehensive is recommended for dataset curation; class2 keeps the old inflammation-first behavior.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: runsTemp/trainval_manual_review_package_YYYYMMDD",
    )
    parser.add_argument("--panel-width", type=int, default=512, help="Width of each panel column.")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def image_files(path):
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def mask_files(path):
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in MASK_SUFFIXES)


def find_mask(mask_dir, image_stem):
    for suffix in (".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = mask_dir / f"{image_stem}{suffix}"
        if path.exists():
            return path
    return None


def load_mask(mask_path):
    if mask_path is None:
        return None
    if mask_path.suffix.lower() == ".npz":
        data = np.load(mask_path)
        if "mask" in data:
            mask = data["mask"]
        else:
            first_key = list(data.keys())[0]
            mask = data[first_key]
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None

    if mask.ndim == 2:
        channels = np.zeros((mask.shape[0], mask.shape[1], len(CLASS_NAMES)), dtype=np.uint8)
        for class_id in range(len(CLASS_NAMES)):
            channels[:, :, class_id] = (mask == class_id + 1).astype(np.uint8) * 255
        return channels

    if mask.ndim == 3 and mask.shape[2] >= len(CLASS_NAMES):
        return mask[:, :, : len(CLASS_NAMES)]

    return None


def ensure_mask_shape(mask, image_shape):
    height, width = image_shape[:2]
    if mask is None:
        return np.zeros((height, width, len(CLASS_NAMES)), dtype=np.uint8)
    if mask.shape[:2] != (height, width):
        channels = []
        for class_id in range(mask.shape[2]):
            channels.append(cv2.resize(mask[:, :, class_id], (width, height), interpolation=cv2.INTER_NEAREST))
        mask = np.stack(channels, axis=-1)
    return (mask > 0).astype(np.uint8) * 255


def resize_to_width(image, width):
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_color_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        class_pixels = mask[:, :, class_id] > 0
        color_mask[class_pixels] = color
    return color_mask


def make_overlay(image, mask):
    color_mask = make_color_mask(mask)
    overlay = image.copy()
    positive = np.any(mask > 0, axis=2)
    blended = cv2.addWeighted(image, 0.62, color_mask, 0.38, 0)
    overlay[positive] = blended[positive]

    for class_id, color in enumerate(CLASS_COLORS):
        binary = (mask[:, :, class_id] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def put_label(panel, text):
    bar = np.full((34, panel.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def make_panel(image, mask, title, width):
    original = resize_to_width(image, width)
    overlay = resize_to_width(make_overlay(image, mask), width)
    color_mask = resize_to_width(make_color_mask(mask), width)

    max_height = max(original.shape[0], overlay.shape[0], color_mask.shape[0])
    panels = []
    for label, panel in (("Original", original), ("GT overlay", overlay), ("GT mask", color_mask)):
        if panel.shape[0] < max_height:
            pad = np.full((max_height - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        panels.append(put_label(panel, label))

    gap = np.full((panels[0].shape[0], 8, 3), 255, dtype=np.uint8)
    body = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    header = np.full((42, body.shape[1], 3), 235, dtype=np.uint8)
    cv2.putText(header, title[:180], (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def class_stats(mask):
    total = float(mask.shape[0] * mask.shape[1])
    pixels = [int(np.count_nonzero(mask[:, :, idx] > 0)) for idx in range(len(CLASS_NAMES))]
    ratios = [value / total for value in pixels]
    present = [CLASS_NAMES[idx] for idx, value in enumerate(pixels) if value > 0]
    present_cn = [CLASS_NAMES_CN[idx] for idx, value in enumerate(pixels) if value > 0]
    return pixels, ratios, present, present_cn


def component_stats(mask):
    total_components = 0
    small_components = 0
    max_components_per_class = 0
    largest_component_ratios = []

    for class_id in range(len(CLASS_NAMES)):
        binary = (mask[:, :, class_id] > 0).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            largest_component_ratios.append(0.0)
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        class_components = max(0, num_labels - 1)
        total_components += class_components
        max_components_per_class = max(max_components_per_class, class_components)

        areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.asarray([], dtype=np.int32)
        if len(areas) > 0:
            small_components += int(np.count_nonzero(areas < 32))
            largest_component_ratios.append(float(np.max(areas)) / float(np.sum(areas)))
        else:
            largest_component_ratios.append(0.0)

    return {
        "component_total": int(total_components),
        "small_component_count": int(small_components),
        "max_components_per_class": int(max_components_per_class),
        "largest_component_ratio_min": float(min(largest_component_ratios)) if largest_component_ratios else 0.0,
    }


def priority_hint_class2(pixels, ratios, has_mask):
    if not has_mask:
        return "高", 100, "缺少mask文件"
    if sum(pixels) == 0:
        return "中", 35, "空mask/阴性样本，确认是否确实无病变"
    if pixels[2] > 0:
        return "高", 70, "包含炎症class2，优先确认是否为成片炎细胞浸润"
    tiny_classes = [CLASS_NAMES_CN[idx] for idx, ratio in enumerate(ratios) if 0 < ratio < 0.001]
    if tiny_classes:
        return "高", 60, "标注区域很小: " + "/".join(tiny_classes)
    if sum(1 for value in pixels if value > 0) >= 2:
        return "中", 35, "多类别共存，检查类别混淆和漏标"
    return "低", 10, "单类别且面积不极端"


def priority_hint_comprehensive(pixels, ratios, components, has_mask):
    score = 0
    reasons = []
    total_pixels = sum(pixels)
    total_ratio = sum(ratios)
    present_count = sum(1 for value in pixels if value > 0)

    if not has_mask:
        return "高", 100, "缺少mask文件"

    if total_pixels == 0:
        score += 45
        reasons.append("空mask/阴性样本，需确认是否确实无病变")

    if 0 < total_ratio < 0.003:
        score += 35
        reasons.append("总标注面积极小")
    elif total_ratio > 0.40:
        score += 30
        reasons.append("总标注面积过大，检查是否过标")

    tiny_classes = [CLASS_NAMES_CN[idx] for idx, ratio in enumerate(ratios) if 0 < ratio < 0.001]
    if tiny_classes:
        score += min(40, 18 * len(tiny_classes))
        reasons.append("类别标注面积很小: " + "/".join(tiny_classes))

    huge_classes = [CLASS_NAMES_CN[idx] for idx, ratio in enumerate(ratios) if ratio > 0.25]
    if huge_classes:
        score += 20
        reasons.append("单类面积偏大: " + "/".join(huge_classes))

    if present_count >= 3:
        score += 25
        reasons.append("三类及以上共存，优先检查类别混淆/漏标")
    elif present_count == 2:
        score += 12
        reasons.append("双类别共存，检查边界和类别")

    if components["component_total"] >= 20:
        score += 25
        reasons.append("连通区域很多，疑似碎片化标注")
    elif components["component_total"] >= 8:
        score += 15
        reasons.append("连通区域偏多")

    if components["small_component_count"] >= 8:
        score += 20
        reasons.append("小碎片标注较多")
    elif components["small_component_count"] >= 3:
        score += 10
        reasons.append("存在多个小碎片标注")

    if present_count > 0 and components["largest_component_ratio_min"] < 0.35:
        score += 10
        reasons.append("至少一个类别缺少主导区域，可能较碎")

    if score >= 55:
        priority = "高"
    elif score >= 25:
        priority = "中"
    else:
        priority = "低"

    if not reasons:
        reasons.append("面积、类别数量、连通区域均无明显异常")
    return priority, int(score), "；".join(reasons)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(dataset_root, splits, output_dir, panel_width, jpeg_quality, priority_strategy):
    dataset_root = Path(dataset_root)
    dataset_name = dataset_root.name
    all_rows = []
    unmatched_rows = []
    summary = {}

    for split in splits:
        images_dir = dataset_root / split / "images"
        masks_dir = dataset_root / split / "masks"
        images = image_files(images_dir)
        masks = mask_files(masks_dir)
        image_stems = {path.stem for path in images}
        mask_stems = {path.stem for path in masks}

        for stem in sorted(mask_stems - image_stems):
            unmatched_rows.append({
                "dataset": dataset_name,
                "split": split,
                "type": "mask_without_image",
                "stem": stem,
                "path": str((masks_dir / f"{stem}.npz").resolve()) if (masks_dir / f"{stem}.npz").exists() else "",
            })

        review_dir = output_dir / "review_images" / dataset_name / split
        review_dir.mkdir(parents=True, exist_ok=True)

        split_rows = []
        for order, image_path in enumerate(tqdm(images, desc=f"{dataset_name}/{split}"), start=1):
            mask_path = find_mask(masks_dir, image_path.stem)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                unmatched_rows.append({
                    "dataset": dataset_name,
                    "split": split,
                    "type": "invalid_image",
                    "stem": image_path.stem,
                    "path": str(image_path.resolve()),
                })
                continue

            raw_mask = load_mask(mask_path)
            mask = ensure_mask_shape(raw_mask, image.shape)
            pixels, ratios, present, present_cn = class_stats(mask)
            components = component_stats(mask)
            if priority_strategy == "class2":
                priority, priority_score, priority_reason = priority_hint_class2(pixels, ratios, mask_path is not None)
            else:
                priority, priority_score, priority_reason = priority_hint_comprehensive(
                    pixels, ratios, components, mask_path is not None
                )

            panel_name = f"{order:05d}_{image_path.stem[:120]}.jpg"
            panel_path = review_dir / panel_name
            if not panel_path.exists():
                title = f"{dataset_name} | {split} | {order:05d} | classes: {','.join(present) if present else 'none'}"
                panel = make_panel(image, mask, title, panel_width)
                cv2.imwrite(str(panel_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

            row = {
                "review_id": f"{dataset_name}_{split}_{order:05d}",
                "dataset": dataset_name,
                "split": split,
                "order": order,
                "auto_priority": priority,
                "auto_priority_score": priority_score,
                "auto_priority_reason": priority_reason,
                "component_total": components["component_total"],
                "small_component_count": components["small_component_count"],
                "max_components_per_class": components["max_components_per_class"],
                "largest_component_ratio_min": f"{components['largest_component_ratio_min']:.6f}",
                "review_image": str(panel_path.relative_to(output_dir)).replace("\\", "/"),
                "image_file": str(image_path.resolve()),
                "mask_file": str(mask_path.resolve()) if mask_path else "",
                "image_stem": image_path.stem,
                "image_width": image.shape[1],
                "image_height": image.shape[0],
                "classes_present": "|".join(present),
                "classes_present_cn": "|".join(present_cn),
                "ballooning_pixels": pixels[0],
                "fibrosis_pixels": pixels[1],
                "inflammation_pixels": pixels[2],
                "steatosis_pixels": pixels[3],
                "ballooning_ratio": f"{ratios[0]:.8f}",
                "fibrosis_ratio": f"{ratios[1]:.8f}",
                "inflammation_ratio": f"{ratios[2]:.8f}",
                "steatosis_ratio": f"{ratios[3]:.8f}",
                "manual_priority": "",
                "label_confidence_1_to_5": "",
                "usable_for_train": "",
                "usable_for_valid": "",
                "needs_relabel": "",
                "suggest_exclude": "",
                "error_type": "",
                "reviewer": "",
                "notes": "",
            }
            split_rows.append(row)

        split_csv = output_dir / f"manual_review_sheet_{dataset_name}_{split}.csv"
        all_rows.extend(split_rows)
        summary[f"{dataset_name}/{split}"] = {
            "images": len(images),
            "masks": len(masks),
            "matched_review_rows": len(split_rows),
            "mask_without_image": len(mask_stems - image_stems),
            "image_without_mask": len(image_stems - mask_stems),
        }

        fieldnames = list(split_rows[0].keys()) if split_rows else []
        if fieldnames:
            write_csv(split_csv, split_rows, fieldnames)

    return all_rows, unmatched_rows, summary


def write_readme(output_dir, summary):
    readme = f"""# Valid/Test 人工审查包

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 目的

这个文件夹用于人工检查 `valid` 和 `test` 的 GT 标注质量，重点判断哪些切片适合作为较可信验证/测试集，哪些切片需要重标、降权或剔除。

本包采用“综合优先级”，不再把 class2/炎症作为唯一优先标准。排序综合考虑：

- 是否为 test/valid：test 更影响最终结论。
- 标注面积是否过小或过大。
- 是否多类别共存。
- 连通区域是否过多，是否存在很多小碎片。
- 是否为空 mask 或缺少 mask。
- 标注形态是否可能碎片化、不典型。

## 文件结构

```text
trainval_manual_review_package_*/
  review_images/
    20260204111923/
      train/
      valid/
    385-liver.groupclean.v1/
      train/
      valid/
  manual_review_sheet.csv
  manual_review_sheet_priority_first.csv
  manual_review_sheet_数据集_split.csv
  unmatched_files.csv
  summary.json
  README_manual_review.md
```

## 图片内容

每张审查图从左到右为：

```text
Original / GT overlay / GT mask
```

- `Original`：原始病理图。
- `GT overlay`：原图上叠加人工标注区域，边界会加粗，最适合判断标注是否合理。
- `GT mask`：纯色 mask，方便快速确认类别和面积。

颜色映射：

```text
class0 ballooning / 气球样变：红色
class1 fibrosis / 纤维化：绿色
class2 inflammation / 炎症：蓝色
class3 steatosis / 脂肪变：黄色
```

## 命名方式

审查图命名格式：

```text
00001_原始图片stem.jpg
```

CSV 中的 `review_id` 格式：

```text
数据集名_split_序号
```

例如：

```text
20260204111923_train_00001
385-liver.groupclean.v1_valid_00007
```

## CSV 填写建议

推荐主要填写这些列：

```text
manual_priority
label_confidence_1_to_5
usable_for_train
usable_for_valid
needs_relabel
suggest_exclude
error_type
reviewer
notes
```

字段含义：

- `manual_priority`：人工优先级，建议填 `高 / 中 / 低`。炎症、漏标明显、边界混乱、多类别混合的图优先级高。
- `label_confidence_1_to_5`：标注可信度。`5` 很可信，`3` 存疑但可参考，`1` 明显错误。
- `usable_for_train`：是否可进训练集，建议填 `是 / 否 / 存疑`。
- `usable_for_valid`：是否可进高可信验证集。验证集要求更高，只有标注清楚、类别典型的图建议填 `是`。
- `needs_relabel`：是否建议重标，填 `是 / 否`。
- `suggest_exclude`：是否建议剔除，填 `是 / 否`。
- `error_type`：可填 `漏标 / 过标 / 类别错误 / 边界粗糙 / 病变不典型 / 阴性样本 / 其他`。
- `notes`：自由备注。

## 审查优先级

脚本已经给出 `auto_priority` 和 `auto_priority_reason`，它只是机器提示，不等于最终结论。建议人工先看：

1. 优先打开 `manual_review_sheet_priority_first.csv`，它已经按 `test > valid > 综合风险分` 排序。
2. 先看 `auto_priority=高` 且 `auto_priority_score` 高的图。
3. test 集优先于 valid 集，因为 test 影响最终模型结论。
4. 多类别共存、标注面积极小/过大、空 mask、小碎片很多的图优先。
5. 炎症仍需要关注，但不再作为唯一优先级标准。

## 当前统计

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""
    (output_dir / "README_manual_review.md").write_text(readme, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("runsTemp") / f"trainval_manual_review_package_{datetime.now().strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    unmatched_rows = []
    summary = {}

    for dataset_root in args.dataset_roots:
        rows, unmatched, dataset_summary = process_dataset(
            dataset_root=dataset_root,
            splits=args.splits,
            output_dir=output_dir,
            panel_width=args.panel_width,
            jpeg_quality=args.jpeg_quality,
            priority_strategy=args.priority_strategy,
        )
        all_rows.extend(rows)
        unmatched_rows.extend(unmatched)
        summary.update(dataset_summary)

    if all_rows:
        write_csv(output_dir / "manual_review_sheet.csv", all_rows, list(all_rows[0].keys()))
        priority_rank = {"高": 0, "中": 1, "低": 2}
        split_rank = {"test": 0, "valid": 1, "train": 2}
        priority_rows = sorted(
            all_rows,
            key=lambda row: (
                split_rank.get(row["split"], 9),
                priority_rank.get(row["auto_priority"], 9),
                -int(row.get("auto_priority_score", 0)),
                row["dataset"],
                int(row["order"]),
            ),
        )
        write_csv(output_dir / "manual_review_sheet_priority_first.csv", priority_rows, list(all_rows[0].keys()))

    unmatched_fields = ["dataset", "split", "type", "stem", "path"]
    write_csv(output_dir / "unmatched_files.csv", unmatched_rows, unmatched_fields)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "output_dir": str(output_dir.resolve()),
        "datasets": args.dataset_roots,
        "splits": args.splits,
        "priority_strategy": args.priority_strategy,
        "total_review_rows": len(all_rows),
        "unmatched_rows": len(unmatched_rows),
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, summary_payload)

    print(f"Manual review package created: {output_dir.resolve()}")
    print(f"Review rows: {len(all_rows)}")
    print(f"Unmatched rows: {len(unmatched_rows)}")
    print(f"Review CSV: {(output_dir / 'manual_review_sheet.csv').resolve()}")
    print(f"README: {(output_dir / 'README_manual_review.md').resolve()}")


if __name__ == "__main__":
    main()
