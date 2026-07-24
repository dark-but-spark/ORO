# MultiResUNet 训练日志分析

> 分析日期: 2026-07-22 | 数据集: 7950 样本 (7155 训练 / 795 验证), 4 类分割任务

---

## 1. 实验概览

| # | 实验名称 | Encoder | cls_w | AugMax | Epochs | Early Stop | 最佳 Val Dice | 最佳 Val Jaccard | 最佳 Epoch | Train-Val Gap |
|---|----------|---------|-------|--------|--------|------------|:----------:|:------------:|:---------:|:-----------:|
| 1 | `cls2w10_os15_20260720_214610` | resnet34 | [1,1,1,1] | 0.4 | 110 | 否 (满) | 0.7623 | 0.6733 | 92 | 0.1073 |
| 2 | `cls2w10_os15_20260721_092621` | resnet34 | [1,1,1,1] | 0.4 | 110 | 否 (满) | 0.7592 | 0.6718 | 108 | 0.1221 |
| 3 | `cls2w125_os15_aug05_*` | resnet34 | [1,1,1.25,1] | **0.5** | 110 | 否 (满) | 0.7649 | 0.6769 | 94 | 0.1024 |
| 4 | `cls2w125_os15_long140_tmax100_*` | resnet34 | [1,1,1.25,1] | 0.4 | 140 | **是** (133) | **0.7685** | 0.6804 | 98 | 0.1099 |
| 5 | `cls2w125_os15_tta_20260719_221021` | resnet34 | [1,1,1.25,1] | 0.4 | 110 | 否 (满) | 0.7680 | **0.6821** | 91 | 0.0998 |
| 6 | `cls2w125_os15_tta_20260720_110418` | resnet34 | [1,1,1.25,1] | 0.4 | 110 | **是** (44) | 0.7630 | 0.6704 | 44 | 0.0309 |
| 7 | `cls2w125_os15_wd1e3_*` | resnet34 | [1,1,1.25,1] | 0.4 | 110 | 否 (满) | 0.7638 | 0.6721 | 100 | **0.0755** |
| 8 | `resnet50_cls2w125_os15_*` | **resnet50** | [1,1,1.25,1] | 0.4 | 110 | 否 (满) | 0.7583 | 0.6705 | 108 | 0.0939 |

> **Legend**: cls_w = 类别权重 (4 类), "满" = 跑满全 epoch 未触发 early stop  
> 命名推导: `P_smp_{encoder}_cls2w{class2_weight*100}_os{oversample_factor*10}_{aug?}_{tta?}_{long?}_{wd?}_{timestamp}`  
> 所有实验共用: lr=2e-5, cosine scheduler, batch_size=16, weight_decay=5e-4 (除 #7)

---

## 2. 最佳模型: `cls2w125_os15_long140_tmax100` (Experiment #4)

**这是当前最优模型。**

- **最佳 Val Dice: 0.7685** (Epoch 98/140)
- **最佳 Val Jaccard: 0.6804**
- 训练 133 epoch 后触发 early stop (patience=35, min_epochs=90)
- 此时 train Dice=0.8621, val Dice=0.7584，说明训练后期 train Dice 显著下降（过拟合后模型退化）

**关键配置差异**:
- 140 epoch + `lr_cosine_t_max=100`（比默认值更大，余弦退火周期更长）
- class_weights=[1.0, 1.0, 1.25, 1.0]（给第 3 类 25% 额外权重）
- oversample_factor=1.5 对类别 2 进行过采样

---

## 3. 亚军: `cls2w125_os15_tta` (Experiment #5)

- **最佳 Val Dice: 0.7680** (Epoch 91/110)
- **最佳 Val Jaccard: 0.6821** ← 所有实验中 Jaccard 最高
- 使用 **validation TTA (flips)**，在验证时做 flip 增强推理
- Train-Val Gap 仅 0.0998，泛化能力好

> 注意: 重复实验 (#6) 在第 44 epoch 就 early stop (Val Dice 0.7630)，说明 TTA 本身可能带来验证集评估的噪声（TTA 增强的 val 评估不稳定），重复性需要进一步验证。

---

## 4. 消融分析

### 4.1 类别权重 (class_weights)

| 实验 | class_weights | Best Val Dice | 差距 |
|------|:-------------:|:------------:|:----:|
| #1 baseline | [1,1,1,1] | 0.7623 | — |
| #2 baseline 重复 | [1,1,1,1] | 0.7592 | — |
| #4 best | [1,1,1.25,1] | 0.7685 | +0.0062 |

**结论**: 给第 3 类 (索引 2) 增加 25% 权重带来约 **+0.6pp** 的 Val Dice 提升，说明该类在数据集中可能欠代表或更难分割。

### 4.2 数据增强强度 (augmentation max level)

| 实验 | Aug Max Level | Best Val Dice |
|------|:------------:|:------------:|
| #4 (long) | 0.4 | 0.7685 |
| #3 (aug05) | **0.5** | 0.7649 |

**结论**: 将 augmentation max level 从 0.4 提升到 0.5 **略微降低了性能** (-0.0036)。0.4 的增强强度可能已经是当前数据集的最优点。

### 4.3 训练时长 (epochs)

| 实验 | Epochs | 实际 Epochs | Best Val Dice |
|------|:------:|:---------:|:------------:|
| #5 | 110 | 110 | 0.7680 |
| #4 | **140** | 133 (ES) | **0.7685** |

**结论**: 更长的训练几乎没带来提升 (+0.0005)。教育成本/收益比不高。最佳模型出现在 epoch 91-98 之间，110 epoch 已足够。

### 4.4 Weight Decay 正则化

| 实验 | Weight Decay | Best Val Dice | Train-Val Gap |
|------|:----------:|:------------:|:------------:|
| #4 | 5e-4 | 0.7685 | 0.1099 |
| #7 | **1e-3** | 0.7638 | **0.0755** |

**结论**: 更高 weight decay (1e-3) 虽然**牺牲了约 0.5pp 的 Dice**，但大幅缩小了 train-val gap（从 0.11 → 0.076），说明有效抑制了过拟合。可能适合需要更好泛化的场景。

### 4.5 Encoder 选择

| 实验 | Encoder | Params | Batch | Best Val Dice |
|------|:-------:|:------:|:-----:|:------------:|
| #4 | **resnet34** | 24.4M | 16 | **0.7685** |
| #8 | resnet50 | 32.5M | 12 | 0.7583 |

**结论**: ResNet50 虽然参数量更大 (32.5M vs 24.4M)，**Val Dice 反而低了约 1pp**。可能原因：
- batch_size 被迫降至 12（显存限制）
- 更大的模型在当前数据规模下更容易过拟合
- ResNet34 对本任务已经是足够的容量

---

## 5. 训练曲线趋势（来自 training.log）

### 典型趋势（以 #8 ResNet50 为例，因 log 最完整）:

- Epoch 1-10: 快速上升期，Val Dice 从 0.49 → 0.72
- Epoch 10-30: 平台期，Val Dice 在 0.72-0.74 震荡
- Epoch 30-40: augmentation curriculum 启动 (cosine ramp)，Dice 继续微升
- Epoch 40-70: 缓慢爬升期，Val Dice 从 0.74 → 0.75
- Epoch 70-110: 微调期，train Dice 继续上升 (0.80 → 0.85) 但 val Dice 几乎不动，gap 扩大

**共同模式**: 所有实验的 Train-Val Dice Gap 在后期都显著扩大（过拟合）。例如实验 #1 从 best epoch 的 0.107 到 final 的 0.122。

---

## 6. 问题与建议

### 问题

1. **过拟合**: Train-Val Dice Gap 在后期普遍达到 0.10-0.12，train Dice 能到 0.88 但 val Dice 卡在 0.76
2. **Val Dice 天花板**: 所有 resnet34 实验的 Val Dice 都卡在 0.76-0.77 附近
3. **训练不稳定性**: 实验 #6 (TTA 重复) 在第 44 epoch 就 early stop，TTA 验证可能引入额外方差
4. **所有实验的 training.err 中都有网络不可达警告**，但未影响训练（权重回退到原始 URL 加载）

### 建议

| 优先级 | 建议 | 预期收益 |
|:----:|------|:------:|
| 🔴 高 | 增加 Dropout/正则化强度，当前可能不足 | 缩小 train-val gap |
| 🔴 高 | 尝试更复杂的数据增强策略（如 MixUp, CutMix） | 提升泛化 |
| 🟡 中 | 尝试 Focal Loss (当前 use_focal_loss=False) | 改善难分类样本 |
| 🟡 中 | 尝试更大的图片分辨率（取消或减小 scale_factor 0.75） | 更多细节信息 |
| 🟢 低 | 尝试更多 epoch + 更强的 early stop patience | label 更充分收敛 |

---

## 7. 所有实验汇总（排序: Best Val Dice ↓）

| Rank | 实验名 | Best Val Dice | Best Val Jaccard | Best Epoch | Total Epochs | Key Config Diff |
|:----:|--------|:-----------:|:--------------:|:--------:|:----------:|-----------------|
| 1 | `cls2w125_os15_long140_tmax100` | **0.7685** | 0.6804 | 98 | 133 (ES) | 140ep, t_max=100 |
| 2 | `cls2w125_os15_tta_221021` | 0.7680 | **0.6821** | 91 | 110 | TTA=flips |
| 3 | `cls2w125_os15_aug05` | 0.7649 | 0.6769 | 94 | 110 | aug_max=0.5 |
| 4 | `cls2w125_os15_wd1e3` | 0.7638 | 0.6721 | 100 | 110 | wd=1e-3 |
| 5 | `cls2w125_os15_tta_110418` | 0.7630 | 0.6704 | 44 | 70 (ES) | TTA, ES@44 |
| 6 | `cls2w10_os15_214610` (baseline) | 0.7623 | 0.6733 | 92 | 110 | cls_w=[1,1,1,1] |
| 7 | `cls2w10_os15_092621` (baseline #2) | 0.7592 | 0.6718 | 108 | 110 | cls_w=[1,1,1,1] |
| 8 | `resnet50_cls2w125_os15` | 0.7583 | 0.6705 | 108 | 110 | resnet50 |

---

*报告由训练日志自动生成，分析范围: `runs/P_*` 目录下 8 个实验。*
