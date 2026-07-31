# 下一轮人工筛选计划

更新日期：2026-07-31

## 一、先给结论

不建议逐张重审 `20260204111923` 的全部 3976 张图片，也不建议把 valid/test 缩成一个只包含“容易样本”的小集合。

推荐方案：

| 数据 | 当前规模 | 人工任务 | 是否全部审查 |
|---|---:|---|---|
| A train | 2782 | 只审程序异常、长期 worst case、疑似漏标样本 | 否 |
| A valid | 794 | 审 150-200 张高风险与分层随机样本，确认问题后修正或排除 | 否 |
| A test | 400 | 补完 worst cases，并完整审查一个 150-200 张的代表性冻结子集 | 部分 |
| B_clean valid | 85 | 全部审查 | 是 |
| B_clean test | 82 | 全部审查 | 是 |
| B_clean train | 671 | 只审异常与抽样样本 | 否 |

A valid 不应只保留 150 张作为唯一验证集。正确做法是用这 150-200 张发现系统性问题，修正或排除确认错误的样本，其余未发现问题的 valid 仍保留。过小的 valid 会使早停和模型排名波动过大。

## 二、A 数据怎么筛

### A valid

建议审查 180 张：

1. 当前最佳模型 valid Dice 最差的 80 张。
2. class2 Dice 最差的 40 张。
3. 每个类别随机抽样 10-15 张，共约 40 张。
4. 完全随机抽样 20 张，防止只看 worst case 造成偏差。

人工确认后：

- 明确漏标、错标、类别错误：修正 mask；暂时不能修正时从 valid 排除。
- 病变不典型但标注仍合理：保留，并标记 `ambiguous=Y`。
- 仅模型漏检或误检、GT 可靠：必须保留，这是有效难例。
- 相邻切片或同一病例不能跨 train/valid/test。

### A test

当前 curated test 有 305 张，但其中多数是“未进入 worst-case 表”，不等于已经人工确认正确。

建议建立一个真正冻结的 `A_test_audited_v1`：

1. 从 400 张中按类别阳性比例和病例分层抽 150-200 张。
2. 这批图片全部人工检查 GT，而不是只检查模型失败区域。
3. 至少 20% 由第二位审查者复核。
4. 完成后冻结文件清单和 mask 版本。
5. 后续所有调参只看 valid；冻结 test 仅在阶段性模型完成后评估。

不要只从“模型表现好”的图片里组成 test，也不要因为模型预测与 GT 不一致就默认删除样本。

## 三、B_clean 怎么筛

`385-liver.groupclean.v1` 的 valid/test 共 167 张，规模可控，建议全部人工审查。

重点确认：

- A、B 两个来源的四个类别定义是否完全一致。
- class2 炎症是否都遵守“成片多炎细胞浸润才算炎症”的标准。
- 标注颜色、通道顺序和类别索引是否一致。
- 同一病例、同一视野或高度相似切片是否仍跨 split。
- 矩形标注是否被错误地当成精确病灶 mask。

只有在标签语义一致、split 无泄漏后，才考虑：

1. B_clean 预训练。
2. 加载 B checkpoint，在 A train 上微调。
3. 仍只用 A valid 选模型，最终分别报告 A test 和 B_clean test。

B_clean 的高分不能替代 A test 分数，也不应与 A test 样本混成一个总分。

## 四、医学生逐张需要判断什么

每张图至少填写：

| 字段 | 填写要求 |
|---|---|
| `gt_reliable` | Y / N / uncertain |
| `error_type` | label_error / model_miss / class_confusion / threshold / tta / none |
| `affected_class` | 0 / 1 / 2 / 3 / multiple |
| `label_issue` | 漏标 / 多标 / 错类 / 边界粗糙 / 病变不典型 / 无 |
| `action` | keep / fix_mask / exclude_eval / second_review |
| `severity` | minor / major |
| `reviewer` | 审查者姓名或编号 |
| `notes` | 简短说明病理依据 |

炎症 class2 的统一口径：

- 单个散在炎细胞一般不标为炎症区域。
- 多个炎细胞形成明确局灶或片状浸润时才标。
- 无法确认时填 `uncertain`，不要直接改成阴性。
- 模型把单个炎细胞周围扩成大片区域，应记为模型误检或边界过扩。
- GT 只标了部分明显浸润区域，应记为 GT 漏标，不应只记模型误检。

## 五、审查后的版本管理

建议目录：

```text
data_review/
  A_valid_review_v1/
    review_sheet.csv
    reviewed_images.txt
    corrected_masks/
    excluded_images.txt
    README.md
  A_test_audited_v1/
    frozen_manifest.csv
    masks/
    README.md
  Bclean_valid_test_review_v1/
    review_sheet.csv
    corrected_masks/
    excluded_images.txt
    README.md
```

每次修改 mask 都记录原文件名、修改原因、审查者和日期。不要覆盖原始数据，使用新版本目录。

## 六、模型增强优先级

### P0：先验证数据清洗是否有效

运行 `temp.sh` 默认四条训练，以两个 seed 比较原始 A 和 filtered A。只有两个 seed 都显示同方向改善，才能认为清洗有效。

### P1：阈值校准

在 valid 上为每个类别独立搜索阈值，例如 0.25-0.75。class2 可以使用更高阈值减少散在假阳性，但阈值必须在 valid 上确定，不能用 test 调。

### P2：病理图像颜色标准化

对 Reinhard/Macenko stain normalization 做单变量实验。先只在 A 内验证；若跨来源 A/B 泛化明显改善，再用于迁移学习。必须保留“无标准化”对照。

### P3：改善小区域与边界

实现并单独比较 Tversky/Focal-Tversky、Boundary loss 或 Dice+BCE+Boundary。每次只增加一种损失，重点看每类 Dice、precision、recall 和小区域召回，不只看总 Dice。

### P4：更合适的预训练

优先尝试病理图像预训练 encoder，其次才是继续增大 ImageNet backbone。必须做随机初始化、ImageNet、病理预训练三组同配置对照。

### P5：高分辨率或 patch 训练

对 class2 和边界细节，可尝试重叠 patch、前景引导采样和多尺度推理。采样概率按“含目标区域”调整，不建议再次仅靠 class2 loss 权重硬推。

### P6：跨来源训练

在 B_clean 完成人工审查前，A 与 B 保持分开。审查完成后采用 `B_clean 预训练 -> A 微调`，不要直接混合随机划分。

## 七、本轮判断标准

filtered 数据版本只有同时满足以下条件才替代原始版本：

1. 两个 seed 的 original test 平均 Dice 不下降。
2. 两个 seed 的 curated test 平均 Dice 上升，或至少更稳定。
3. class0、class1、class3 没有明显牺牲。
4. class2 precision/recall 的变化符合医学目标。
5. B_clean cross-source 结果只作为参考，不参与 A 模型选择。
