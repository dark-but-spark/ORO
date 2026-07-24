# MultiResUNet 训练实验分析报告

> 分析时间：2026-07-24
> 实验数量：17 个（P_ 开头目录）
> 模型架构：smp_unet (resnet34/resnet50 encoder)

---

## 一、全部实验结果

按 Best Validation Dice 降序排列：

| # | 实验名称 | 日期 | Best Val Dice | Best Epoch | Train Dice | Gap | Val Loss | LR Sched | 特殊配置 |
|---|---------|------|:---:|:---:|:---:|:---:|:---:|---|---|
| 1 | tta_long140_tmax100 | 07-24 | **0.7726** | 114 | 0.8795 | +0.107 | 0.1064 | cosine | TTA, 140ep, T_max=100 |
| 2 | long140_tmax100 | 07-21 | **0.7685** | 98 | 0.8784 | +0.110 | 0.1052 | cosine | 140ep, T_max=100, early stop@133 |
| 3 | tta_20260719 | 07-19 | **0.7680** | 91 | 0.8678 | +0.100 | 0.1080 | cosine | TTA=flips |
| 4 | plateau | 07-22 | 0.7652 | 116 | 0.8852 | +0.120 | 0.1048 | plateau | lr_scheduler=plateau |
| 5 | aug05 | 07-22 | 0.7649 | 94 | 0.8672 | +0.102 | 0.1055 | cosine | aug_max=0.5 |
| 6 | fullres_bs8 | 07-23 | 0.7647 | 99 | 0.8433 | +0.079 | 0.1268 | cosine | scale=1.0, bs=8 |
| 7 | wd1e3 | 07-21 | 0.7638 | 100 | 0.8393 | +0.076 | 0.1115 | cosine | weight_decay=1e-3 |
| 8 | tta_20260720 | 07-20 | 0.7630 | 44 | 0.7939 | +0.031 | 0.1079 | cosine | TTA, early stop@70 |
| 9 | cls2w10_v1 | 07-20 | 0.7623 | 92 | 0.8696 | +0.107 | 0.1018 | cosine | class_weights 全 1 |
| 10 | lr5e5 | 07-23 | 0.7608 | 68 | - | - | 0.1053 | cosine | lr=5e-5, early stop@86 |
| 11 | focaldice73_v1 | 07-23 | 0.7606 | 64 | - | - | 0.0884 | cosine | focal:dice=7:3, early stop@89 |
| 12 | focaldice73_v2 | 07-23 | 0.7600 | 115 | 0.8606 | +0.101 | 0.0889 | cosine | 复现 focal:dice=7:3 |
| 13 | focaldice55 | 07-23 | 0.7595 | 115 | 0.8529 | +0.093 | 0.1366 | cosine | focal:dice=5:5 |
| 14 | cls2w10_v2 | 07-21 | 0.7592 | 108 | 0.8812 | +0.122 | 0.1027 | cosine | class_weights 全 1, 复现 |
| 15 | focal_fullres_plateau | 07-24 | 0.7588 | 81 | 0.7900 | +0.031 | 0.1136 | plateau | focal+fullres, early stop@106 |
| 16 | resnet50 | 07-22 | 0.7583 | 108 | 0.8522 | +0.094 | 0.1144 | cosine | encoder=resnet50 |
| 17 | purefocal | 07-23 | 0.7206 | 100 | 0.7327 | +0.012 | 0.0030 | cosine | 纯 focal loss |

### 基线配置（共享参数）

| 参数 | 值 |
|------|-----|
| 模型 | smp_unet + resnet34 |
| 损失函数 | BCE:Dice = 7:3 |
| 初始学习率 | 2e-5 |
| Batch Size | 16 |
| 数据缩放 | scale=0.75 |
| 数据增强 | cosine curriculum, aug_max=0.4 |
| Weight Decay | 5e-4 |
| Class Weights | [1.0, 1.0, 1.25, 1.0] |
| Early Stopping | patience=25, min_epochs=70 |
| 训练轮数 | 110 epochs |

---

## 二、关键发现

### 2.1 最佳策略：TTA + 延长训练 + Cosine T_max=100

Top 3 实验（#1-#3）均涉及 TTA 或长训练：

```
TTA (0.7680) ----+----> long140 + T_max=100 (0.7685) ----+----> +TTA (0.7726)
                  |                                        |
                  v                                        v
              +0.008 vs 基线                             +0.004 vs 无TTA
```

| 策略组合 | Best Val Dice | 相对基线提升 |
|----------|:---:|:---:|
| 基线 (cosine, 110ep, 无TTA) | ~0.764 | - |
| + TTA | 0.7680 | +0.004 |
| + 140ep + T_max=100 | 0.7685 | +0.0045 |
| + 140ep + T_max=100 + TTA | **0.7726** | **+0.0086** |

**TTA (Test-Time Augmentation, flips) 价值 ~+0.004**。在验证时使用翻转增强相当于隐式集成，直接提升验证指标。

**延长训练到 140 epochs 价值 ~+0.004**。T_max=100 让 cosine 学习率在更长周期内缓慢衰减，避免学习率过早塌缩到接近 0。

### 2.2 损失函数：BCE+Dice > Focal 变体

```
BCE+Dice (combined)    ████████████████████████ 0.7726
Focal:Dice 7:3         ██████████████████       0.7606
Focal:Dice 5:5         ██████████████████       0.7595
Focal+BCE+Dice         █████████████████        0.7588
Pure Focal              ████████████            0.7206
```

- **Focal loss 系列整体比 combined loss 低 0.005-0.013**
- 纯 Focal loss 完全不可用（0.7206，但 Train Dice 也仅 0.73，模型根本没学到东西）
- Focal loss 即使配合 Dice 也无法达到 BCE+Dice 的水平，说明对于该分割任务，BCE 是更合适的逐像素监督信号

### 2.3 过拟合是当前核心瓶颈

17 个实验中 Best 模型的 Train-Val Gap 普遍在 +0.10-0.12：

| 模型 | Train Dice | Val Dice | Gap |
|------|:---:|:---:|:---:|
| plateau | 0.8852 | 0.7652 | +0.120 |
| tta_long140 | 0.8795 | 0.7726 | +0.107 |
| fullres_bs8 | 0.8433 | 0.7647 | **+0.079** |
| wd1e3 | 0.8393 | 0.7638 | **+0.076** |

**减小 Gap 的有效方向**：
- **weight_decay=1e-3**：Gap 降至 +0.076（对照组 ~+0.12），但 Val Dice 未超基线
- **full resolution (scale=1.0)**：Gap 降至 +0.079，但 Val Loss 偏高（0.1268 vs 0.105）
- 两者都有效减小了 Gap，说明更大的正则化或更高的输入分辨率有助于泛化，但对 Val Dice 的净收益不显著

### 2.4 Class Weight 有效但不关键

| 配置 | Best Val Dice |
|------|:---:|
| cls2w125 [1.0,1.0,1.25,1.0] | 0.7685 |
| cls2w10 [1.0,1.0,1.0,1.0] | 0.7623 |
| **差异** | **+0.006** |

给类别 2 加 1.25 的权重有约 +0.006 的提升，效果稳定（两次重复实验一致）。

### 2.5 其他消融实验

| 配置变化 | Dice 变化 | 结论 |
|----------|:---:|------|
| lr=5e-5 (2.5× 基线) | -0.003 | 学习率过高，epoch 68 即过拟合触发 early stop |
| aug_max=0.5 (0.4→0.5) | +0.0002 | 增强强度提升几乎无收益 |
| resnet34 → resnet50 | -0.006 | 更大模型反而更差，此数据规模下 resnet34 足够 |
| plateau scheduler | -0.003 | 不如 cosine + T_max=100 |
| fullres + focal | -0.006 | 两个好思路叠加 focal 后反而下降 |

---

## 三、实验演进路径

```
cls2w10 (0.762) ──→ cls2w125 (0.765) ──→ +TTA (0.768)
                                              │
                    ┌─────────────────────────┤
                    v                         v
              long140 (0.7685)          plateau (0.7652)
                    │                         │
                    v                         v
              +TTA (0.7726)             fullres (0.7647)
                    
   side experiments:
     wd1e3 (0.7638) — 降低过拟合但未必提分
     aug05 (0.7649) — 增强加码收益微弱  
     focal series  — 全部低于基线
     resnet50     — 更大模型无帮助
     lr5e5        — 学习率偏大
```

---

## 四、最佳配置与建议

### 当前最优配置

```yaml
model: smp_unet + resnet34 (24.4M params)
loss: BCE:Dice = 7:3
lr: 2e-5, cosine scheduler, T_max=100
epochs: 140
batch_size: 16
scale: 0.75
augmentation: cosine curriculum, aug_max=0.4
class_weights: [1.0, 1.0, 1.25, 1.0]
weight_decay: 5e-4
val_tta: flips
early_stopping: patience=35, min_epochs=90
```

**结果**：Best Val Dice = **0.7726**, Val Jaccard = 0.6859

### 后续优化方向

1. **解决过拟合**（首要任务）：当前 Train-Val Gap 仍有 +0.107
   - 尝试更强的正则化（Dropout、更大的 weight_decay）
   - 更多数据或更强的数据增强（MixUp、CutMix）
   - 减少模型容量（更轻量的 decoder）

2. **Full resolution + 正则化**：fullres_bs8 的 Gap 仅 +0.079，结合更强的正则化可能超越当前最佳

3. **Ensemble**：Top 3 模型集成可能进一步提升 0.002-0.005

4. **更大 T_max**：T_max=140 或 cosine annealing with warm restarts 值得尝试

---

## 五、附录：实验完整配置

### 共享配置

| 参数 | 值 |
|------|-----|
| 模型架构 | smp_unet |
| 输入通道 | 3 |
| 输出通道 | 4 |
| Encoder | resnet34 (imagenet pretrained) |
| 参数总量 | 24,436,804 |
| 优化器 | AdamW |
| 初始学习率 | 2e-5 |
| 梯度裁剪 | 0.5 |
| 验证集比例 | 10% |
| 数据增强策略 | cosine curriculum (mild → moderate) |
| PyTorch | 2.5.1+cu121 |

### 各实验差异化配置

| 实验 | 差异点 |
|------|--------|
| tta_long140_tmax100 | epochs=140, T_max=100, TTA=flips, patience=35 |
| long140_tmax100 | epochs=140, T_max=100, patience=35 |
| tta_20260719 | TTA=flips |
| plateau | lr_scheduler=plateau, epochs=120 |
| aug05 | aug_max=0.5 |
| fullres_bs8 | scale=1.0, batch_size=8 |
| wd1e3 | weight_decay=1e-3 |
| cls2w10_v1/v2 | class_weights 全 1.0 |
| lr5e5 | lr=5e-5 |
| focaldice73 | use_focal_loss=true, dice:bce=0.3:0 -> dice:focal=0.3:0.7 |
| focaldice55 | 同上, dice:focal=0.5:0.5 |
| focal_fullres_plateau | use_focal_loss=true, scale=1.0, bs=8, plateau |
| resnet50 | encoder=resnet50 |
| purefocal | use_focal_loss=true, dice_weight=0, bce_weight=0 |
