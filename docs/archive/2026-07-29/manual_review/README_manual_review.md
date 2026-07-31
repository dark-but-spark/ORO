# 人工复核包说明

本复核包由以下机器诊断结果整理得到：

```text
E:\project\ORO\runsTemp\diagnostics\best_20260724_tta
E:\project\ORO\runsTemp\diagnostics\best_20260724_no_tta
```

目的不是重新计算模型分数，而是请人工判断：

```text
当模型预测和人工标注不一致时，问题主要来自哪里？
```

## 文件内容

```text
manual_review_sheet.csv       人工复核填写表格，建议用 Excel/WPS 打开
tta_vs_no_tta_report.csv      全部 795 张验证图的 TTA 与 no-TTA 机器对比结果
summary.json                  机器统计摘要
images/01_tta_worst           TTA 模式下最差的 50 个样本
images/02_tta_harmful         TTA 明显比 no-TTA 更差的样本
images/03_class2_hard         class2 困难样本
```

建议优先查看：

```text
images/01_tta_worst
```

这些是当前模型得分最低、最影响整体指标的样本。

## 每张图片怎么看

每个 JPG 是横向拼接图，从左到右依次为：

```text
image          原始组织图
ground_truth   人工标注，也就是 GT
prediction     模型二值预测结果
probability    模型概率图，越亮表示模型越倾向于认为该处属于目标
error map      错误图：红色=模型多预测，绿色=模型漏预测
gt_overlay     人工标注叠加在原图上
pred_overlay   模型预测叠加在原图上
```

复核时重点看：

```text
1. ground_truth 是否标在正确的组织结构上
2. prediction 是否落在医学上合理的结构上
3. gt_overlay 和 pred_overlay 谁更符合原图
4. probability 是否已经在正确位置发亮
5. no-TTA 是否比 TTA 明显更好
```

## 人工错误类型

请在 `manual_review_sheet.csv` 的 `manual_error_type` 列中填写一个主要类型：

```text
A  标注问题
   人工 GT 明显标错、偏移、漏标，或标到了不该标的结构。

B  模型漏检
   人工 GT 可信，但模型没有找到目标；概率图通常也不明显。

C  类别混淆
   模型找到了医学上合理的结构，但预测成了错误类别。

D  TTA 导致变差
   no-TTA 结果还可以，但 TTA 后明显变差。

E  阈值问题
   probability 图在正确位置已经发亮，但最终 prediction 没出来。
   这通常说明 0.5 阈值可能偏高，后续可尝试类别阈值调整。

U  不确定
   无法明确判断，或者需要更高年级/病理老师进一步确认。
```

每张图尽量只选一个主类型。如果确实有多个问题，可以在 `notes` 列补充说明。

## 表格填写字段

主要需要人工填写以下列：

```text
manual_error_type
gt_reliable
prediction_medically_plausible
suggest_fix_label
reviewer
notes
```

建议填写格式：

```text
manual_error_type: A / B / C / D / E / U
gt_reliable: yes / no / uncertain
prediction_medically_plausible: yes / no / uncertain
suggest_fix_label: yes / no
reviewer: 复核人姓名或编号
notes: 简短备注
```

## 推荐复核流程

```text
1. 用 Excel 或 WPS 打开 manual_review_sheet.csv
2. 先查看 images/01_tta_worst 文件夹
3. 从 001 开始逐张看
4. 对照 gt_overlay 和 pred_overlay
5. 判断 GT 和模型预测谁更合理
6. 在表格里填写 manual_error_type 等字段
7. 再查看 images/02_tta_harmful，判断是否需要做 gated TTA
8. 最后查看 images/03_class2_hard，判断 class2 是否需要清洗标注或加强训练样本
```

## 给医学生的说明话术

可以直接这样说明：

```text
我们不是让你评价模型分数，也不是让你改代码。
我们想知道：当模型预测和人工标注不一致时，主要是谁的问题。

请重点看 gt_overlay 和 pred_overlay：
如果人工标注明显不符合原图结构，标 A；
如果人工标注正确，但模型没找到，标 B；
如果模型找到了结构但类别错了，标 C；
如果 no-TTA 还可以但 TTA 变差，标 D；
如果概率图已经找到位置但最终二值结果没出来，标 E；
如果无法判断，标 U。

备注中可以简单写：
“GT 偏移”、“模型漏掉小目标”、“模型标到相似结构”、“类别不确定”、“需要老师确认”等。
```

## 机器摘要

```json
{
  "tta_samples": 795,
  "selected_review_rows": 110,
  "tta_mean_dice": 0.7735128956784302,
  "no_tta_mean_dice": 0.7663322792754418,
  "mean_tta_delta": 0.00718061640298844,
  "tta_improved_count": 501,
  "tta_worse_count": 282,
  "tta_severe_harm_count_delta_lt_minus_0_2": 10,
  "dice_lt_0_001": 38,
  "dice_lt_0_05": 40,
  "dice_lt_0_2": 50
}
```

## 复核结果怎么用

人工复核完成后，统计 A/B/C/D/E/U 的数量：

```text
A 多：优先修正标注或剔除异常样本
B 多：增加 hard example mining 或补充类似训练样本
C 多：检查类别定义和类别权重
D 多：实现 gated TTA，不对所有样本强制使用 TTA
E 多：做 per-class threshold sweep，尤其关注 class2
U 多：需要更明确的标注规范或更高年级复核
```

最重要的是：机器已经筛出了最值得看的样本，人工现在要判断这些样本为什么错。
