# 2026-07-25 Class2 忽略评分与下一轮训练计划

## 本轮问题

当前最佳模型在原始诊断集上表现已经进入平台期，但错误集显示 class2 存在较多标注不全、弱病变边界不明确、以及疑似非显著病变的问题。因此本轮先回答两个问题：

1. 如果只在评分阶段忽略 class2，当前模型能到什么水平。
2. 是否应该通过小训练降低 class2 的损失权重，验证 class2 噪声是否正在拖累共享特征。

## 已新增代码

`analyze_ignore_class.py`

用途：

- 读取 `runsTemp/diagnostics/*/per_class_summary.json`
- 读取 `runsTemp/diagnostics/*/validation_error_report.csv`
- 在不重新推理、不改标签、不改模型的前提下，重新计算忽略指定类别后的分数
- 默认忽略 `class2`

推荐运行：

```bash
python analyze_ignore_class.py \
  --diagnostics-dir runsTemp/diagnostics \
  --ignore-classes 2 \
  --output-dir runsTemp/diagnostics_ignore_class_analysis
```

## 当前忽略 class2 的结果

| 诊断版本 | 原始 overall Dice | 忽略 class2 后 Macro Dice | 忽略 class2 后 Micro Dice |
| --- | ---: | ---: | ---: |
| best_20260724_tta | 0.7735 | 0.9499 | 0.8449 |
| best_20260724_no_tta | 0.7663 | 0.9464 | 0.8362 |

解释：

- `Macro Dice` 是 class0/class1/class3 的类别均值，容易被空类别和简单类别抬高。
- `Micro Dice` 是汇总 class0/class1/class3 的 TP/FP/FN 后重新计算，更接近忽略 class2 后的真实像素级表现。
- 因此当前更可信的判断是：忽略 class2 后，模型大约在 `0.84` 水平，而不是 `0.95`。

## 下一步机器训练目的

本轮不直接删除 class2，而是用小训练验证：

1. 降低 class2 权重是否能提升整体 Dice。
2. 降低 class2 权重是否能提升 class0/class1/class3 的稳定性。
3. class2 oversampling 是否还需要保留。
4. TTA 是否在模型选择阶段放大或掩盖 class2 问题。

## temp.sh 中的实验设计

固定骨架：

- `smp_unet`
- `resnet34 + imagenet`
- `scale-factor 0.75`
- `combined loss: BCE 0.7 + Dice 0.3`
- `cosine lr, T_max=100`
- `mild -> moderate cosine curriculum`
- `checkpoint-interval 0`
- `tb-image-interval 0`
- `metric-ignore-classes 2`

新增训练记录指标：

- `Metrics/val_dice_ignore_classes_2`
- `Metrics/val_jaccard_ignore_classes_2`
- `Metrics/val_macro_class_dice_ignore_classes_2`
- `Metrics/val_macro_class_jaccard_ignore_classes_2`

说明：原始 `Metrics/val_dice` 仍然包含 class2，并继续用于历史对比和默认 best model 选择。新增指标只用于观察“去掉 class2 后”的模型能力。

实验：

| 实验名 | 变量 | 目的 |
| --- | --- | --- |
| Q_cls2w125_os15_tta_anchor | class2 weight 1.25, oversample 1.5, TTA | 本批次锚点 |
| Q_cls2w10_os15_tta | class2 weight 1.0 | 去掉 class2 额外加权 |
| Q_cls2w075_os15_tta | class2 weight 0.75 | 轻度降低 class2 监督 |
| Q_cls2w05_os15_tta | class2 weight 0.5 | 明显降低 class2 监督 |
| Q_cls2w075_os12_tta | class2 weight 0.75, oversample 1.2 | 分离 loss 权重和 oversample 的影响 |
| Q_cls2w10_noos_tta | class2 weight 1.0, no oversample | 测试 oversample 是否放大噪声 |
| Q_cls2w075_os15_no_tta | class2 weight 0.75, no TTA validation | 查看模型本体能力 |

## 判断标准

每条实验结束后优先看：

1. `best_val_dice`
2. `best_val_jaccard`
3. `Metrics/class_0_val_dice`
4. `Metrics/class_1_val_dice`
5. `Metrics/class_2_val_dice`
6. `Metrics/class_3_val_dice`
7. 忽略 class2 后的 `micro_without_ignored_dice`
8. TTA 与 no-TTA 差异

如果 `class2 weight 0.75/0.5` 让 overall 或 class0/1/3 上升，同时 class2 下降不多，说明 class2 噪声确实在拖共享特征。

如果 class2 降权后 overall 不升反降，说明 class2 虽然噪声多，但仍提供了重要结构监督，下一步应转向人工高置信验证集和 class2 置信 mask，而不是继续降权。

## 人工验证并行建议

机器训练期间，专业人士可以从 worst cases 和 class2 低分样本中选一个高置信小验证集：

- 80 到 150 张
- class2 阳性至少 30 到 50 张
- 每张标记 `可靠 / 可疑 / 标注不全 / 不应计入`
- 对 class2 标注不全或定义模糊的样本，不急着删除，先作为单独分组

这个小集合用于回答：模型在“可靠 class2 标注”上到底是真差，还是被原验证集噪声压低。
