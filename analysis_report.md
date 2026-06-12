# MultiResUNet 训练日志分析报告

> 分析日期：2026-06-08
> 数据来源：`logs/logs/` 下 8 组实验 + `training_experiment_log.md`
> 分析范围：阶段 1~7 全部实验结果

---

## 一、当前最佳结果

```text
实验名称:     C_scale075_l04_cls2w15
scale-factor: 0.75
loss:         combined loss, BCE:Dice = 0.7:0.3
augmentation: mild → moderate, cosine curriculum
max_level:    0.4
ramp:         start E40, ramp 35 epochs
class_weight: [1, 1, 1.5, 1]
best val_dice:   0.6836 @ E67
best val_jaccard: 0.5634
```

---

## 二、完整实验演化路径

```
baseline (0.622) → combined loss (0.633) → +mild aug (0.665) → curriculum (0.667)
→ scale 0.75 (0.681) → class2 瓶颈分析 → 当前 8 组实验 → cls2w15 (0.684)
```

### 关键转折点

| 阶段 | 操作 | Val Dice | 增益 |
|:---|:---|:--------:|:----:|
| 阶段 1 | A_plain_baseline | 0.6223 | — |
| 阶段 2 | B_combined_mild_aug_150 | 0.6653 | +0.043 |
| 阶段 3 | curriculum level 扫描 | 0.6673 | +0.002 |
| 阶段 5 | **scale 0.5→0.75** | **0.6814** | **+0.014** |
| 阶段 7 | + class2 weight 1.5 | 0.6836 | +0.002 |

**结论：scale 0.75 是单次最大增益来源，class weight 边际效果有限。**

---

## 三、核心瓶颈：Class 2

### Per-Class Dice（baseline `C_l04_ramp35_scale075`）

| Class | Dice | 评估 |
|:-----:|:----:|:---|
| Class 0 | 0.7685 | 🟢 正常 |
| Class 1 | 0.7298 | 🟢 正常 |
| **Class 2** | **0.4259** | 🔴 **致命短板** |
| Class 3 | 0.7670 | 🟢 正常 |

- Class 2 仅为其他类的 **55%**
- 如果 class 2 提升到 0.65，整体 Dice 理论上可到 ~0.75
- Class 2 低的原因待查：标注质量 / 边界模糊 / 类别混淆

---

## 四、8 组实验完整对比

| # | 实验名称 | Val Dice | Jaccard | Best Epoch | Train-Val Gap | vs Baseline |
|:--:|---------|:--------:|:-------:|:----------:|:-------------:|:-----------:|
| 🥇 | `C_scale075_l04_cls2w15` | **0.6836** | 0.5634 | 67 | -0.030 | baseline |
| 🥈 | `C_scale075_l035_cls2w15` | 0.6815 | 0.5591 | 67 | -0.023 | -0.0021 |
| 🥉 | `C_scale075_l04_cls2w15_dice04` | 0.6802 | 0.5568 | 67 | -0.030 | -0.0034 |
| 4 | `C_scale075_l04_ramp50_cls2w15` | 0.6767 | 0.5587 | 68 | -0.012 | -0.0069 |
| 5 | `C_scale075_l04_cls2w20` | 0.6597 | 0.5334 | 67 | -0.008 | -0.0239 |
| 6 | `C_scale075_l04_cls2w15_lr15e5` | 0.6513 | 0.5236 | 61 | -0.038 | -0.0323 |
| 7 | `C_scale100_mild_lr15e5` | 0.6446 | 0.5166 | 57 | +0.015 | -0.0390 |
| 8 | `C_scale100_l02_ramp45_cls2w15_lr15e5` | 0.6409 | 0.5115 | 57 | +0.016 | -0.0427 |

---

## 五、逐实验解读

### 实验 1：`cls2w15` — 唯一正收益 ✅

- **改动**：class_weight `[1,1,1.5,1]`，class 2 权重 1.5
- **结果**：Val Dice +0.0022（微弱提升）
- **解读**：方向正确但幅度小。需确认 per-class Dice 中 class 2 是否提升，若 class 2 提升了但其他类略降，可继续微调 1.5→1.7

### 实验 2：`cls2w20` — 过度加权反噬 ❌

- **改动**：class_weight `[1,1,2.0,1]`，class 2 权重 2.0
- **结果**：Val Dice -0.0239（显著下降）
- **解读**：2.0 惩罚过重，牺牲了其他类性能。class 2 的 sweet spot 在 1.5–1.8 之间

### 实验 3：`dice04` — Dice Loss 不是瓶颈 ➖

- **改动**：BCE:Dice 从 0.7:0.3 → 0.6:0.4
- **结果**：Val Dice -0.0034（几乎不变）
- **解读**：当前 0.3 的 Dice 权重已足够，边际贡献见顶

### 实验 4：`l035` — 增强上限不敏感 ➖

- **改动**：max_aug_level 0.4 → 0.35
- **结果**：Val Dice -0.0021（基本持平）
- **解读**：0.35–0.4 是安全区间，保留 0.4 即可

### 实验 5：`ramp50` — 慢增强无益 ❌

- **改动**：curriculum_ramp_epochs 35 → 50
- **结果**：Val Dice -0.0069
- **解读**：模型在 ~E40-50 已到最佳增强强度，延长 ramp 让模型在弱增强下待太久

### 实验 6：`lr15e5` — 降低学习率有害 ❌

- **改动**：learning_rate 2e-5 → 1.5e-5
- **结果**：Val Dice -0.0323（**最大降幅之一**）
- **解读**：模型需要更大 lr 跳出局部最优。这个发现打开了尝试更高 lr 的方向

### 实验 7-8：`scale100` — 全分辨率再次不可行 ❌

- **改动**：scale_factor 0.75 → 1.0
- **结果**：两次验证 Dice 均崩塌到 0.64
- **解读**：全分辨率 + 当前架构/数据量 = 过拟合加速。与早期 `scale100_bs8` (0.6417) 一致

---

## 六、参数消融总结

| 参数 | 当前最优 | 测试范围 | 敏感性 | 备注 |
|:---|:---|:---|:---:|:---|
| scale_factor | **0.75** | 0.5–1.0 | 🔴 高 | 单次最大增益来源 |
| learning_rate | **2e-5** | 1.5e-5–2e-5 | 🔴 高 | 3e-5/5e-5 待测 |
| class_weight[2] | **1.5** | 1.0–2.0 | 🟡 中 | 1.7/1.8 待测 |
| batch_size | **16** | 8–24 | 🟡 中 | 24 明显变差 |
| max_aug_level | **0.4** | 0.2–0.6 | 🟢 低 | 0.35–0.4 均可用 |
| bce:dice ratio | **0.7:0.3** | 0.6:0.4 | 🟢 低 | 几乎无影响 |
| ramp_epochs | **35** | 35–50 | 🟢 低 | 35 足够 |
| curriculum 类型 | **cosine fixed** | fixed/adaptive | 🟡 中 | adaptive 未超越 fixed |

---

## 七、前景判断

### 能否达到 0.9+ Dice？

| 问题 | 回答 |
|:---|:---|
| 仅调参能到 0.9？ | **不能。** 当前天花板约 0.72 |
| 不改代码能到 0.9？ | **不能。** 需要架构、损失函数、数据等多方位改动 |
| 改代码后能到 0.9？ | **取决于标注质量。** 如果标注本身的 Dice 上限 < 0.9，模型不可能超越 |
| 务实的目标？ | 调参 → 0.72，+预训练 encoder → 0.76–0.78，+数据改进 → 0.80–0.82 |

### Class 2 为什么只有 0.43？

三类可能（需样本级检查确认）：
1. **标注质量问题**：class 2 标注不一致或存在 label noise
2. **物理边界模糊**：TaOx-MnO2 相变过渡区在像素级别没有清晰边界
3. **类别混淆**：class 2 与 class 3 或 class 1 的视觉特征难以区分

---

## 八、下一步建议（按优先级）

### 🔴 高优先级 — 立即执行

**1. 获取 `cls2w15` 的 per-class Dice**

确认 class 2 是否从 0.426 提升。如果提升了，继续细调 class weight（1.5→1.6→1.7→1.8）。

**2. 样本级预测图检查 class 2**

随机抽 20 张预测结果，检查：
- class 2 的错误集中在哪些位置（边界/内部/与哪些类混淆）
- class 2 的标注是否一致
- class 2 和 class 3 是否存在标注歧义

**3. 尝试更高学习率**

```
C_scale075_l04_cls2w15_lr3e5   # lr=3e-5
C_scale075_l04_cls2w15_lr5e5   # lr=5e-5
```

### 🟡 中优先级 — 小改代码

**4. 更换预训练 Encoder（预估 +0.05–0.10 Dice）**

```python
# 当前：随机初始化 MultiResUNet（7.25M 参数）
# 改为：ImageNet 预训练 backbone
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name='efficientnet-b3',
    encoder_weights='imagenet',
    in_channels=3,
    classes=4,
)
```

**5. 添加 Boundary Loss**

```python
# 对材料界面边界施加额外监督
loss = bce_loss + dice_loss + 0.1 * boundary_loss
```

**6. 测试时增强（TTA）**

```python
# 多尺度 + 翻转推理，预估 +0.02–0.03 Dice
```

### 🟢 低优先级 — 大改动

**7. Class 2 Oversampling**

修改 DataLoader，对包含 class 2 的 patch 增加采样概率。

**8. 尝试 Focal Loss + Class Weight 组合**

虽然早期 focal loss 单独使用很差（0.5239），但与 class weight 组合可能产生协同效应。

---

## 九、已确认不推荐的方向

```text
1. scale=0.5 下继续调增强强度
2. batch_size=24 或更大
3. scale=1.0 + 强课程增强（已验证 3 次均失败）
4. 原始 strict adaptive 参数
5. focal loss 单独路线
6. 继续增加 ramp_epochs
7. 继续增加 dice_weight
```

---

## 十、推荐实验模板

以 baseline config 为基础，每次只改 1 个参数：

```json
{
  "scale_factor": 0.75,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "bce_weight": 0.7,
  "dice_weight": 0.3,
  "class_weights": [1.0, 1.0, 1.5, 1.0],
  "use_combined_loss": true,
  "use_focal_loss": false,
  "augmentation_curriculum": "cosine",
  "curriculum_base_strength": "mild",
  "curriculum_target_strength": "moderate",
  "curriculum_max_aug_level": 0.4,
  "curriculum_start_epoch": 40,
  "curriculum_ramp_epochs": 35,
  "weight_decay": 0.0005,
  "epochs": 130,
  "seed": 42
}
```

---

## 附录：评价标准

每个实验需关注以下维度：

```text
1. best val_dice — 峰值性能
2. best val_jaccard — IoU 指标
3. per-class Dice — 各类别均衡性（尤其是 class 2）
4. train-val gap — 过拟合程度
5. tail10 mean/std — 尾部稳定性
6. best epoch — 收敛速度
```
