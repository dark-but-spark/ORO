# Curated Test Dataset From Manual Review

生成时间：2026-07-31T13:50:31

## 来源

- 原始 test：`E:\project\ORO\data\20260204111923\test`
- 人工审查表：`E:\project\ORO\test_manual_review_sheet.csv`
- 输出目录：`E:\project\ORO\data\20260204111923_curated_manual_review_20260731`

## 判定规则

这是一个相对保守的 curated test。对出现在人工 worst-case 审查表中的图片：

- 任一记录 `suggest_action` 属于 `exclude_eval`，排除。
- 任一记录 `suggest_action` 属于 `fix_label / GT漏标 / GT错标 / GT漏标错标`，排除。
- 任一记录 `gt_reliable = N`，排除。
- 其余人工确认 GT 相对可靠的样本保留。
- 没出现在 worst-case 审查表中的原始 test 样本暂时保留，状态记为 `unreviewed_not_worst`。

## 数量

- 原始 test 图片数：400
- 人工审查记录数：170
- 人工审查涉及去重图片数：119
- curated 保留图片数：305
- 排除图片数：95

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
