# MultiResUNet / SMP-Unet Training Experiment Log

本文档总结截至 2026-06-21 的训练路径、最新运行结果、代码能力更新，以及下一批训练计划。

## 当前最佳结果

当前最高验证 Dice 来自 SMP-Unet ResNet34 + TTA：

```text
实验: P_smp_resnet34_cls2w125_tta
模型: SMP-Unet
encoder: resnet34
encoder_weights: imagenet
scale-factor: 0.75
loss: combined loss, BCE:Dice = 0.7:0.3
class weights: [1, 1, 1.25, 1]
val_tta: flips
best val_dice: 0.76646 @ E40
best val_jaccard: 0.67175
per-class Dice @best: class0=0.8448, class1=0.8007, class2=0.6251, class3=0.8456
final val_dice: 0.75745
tail10 val_dice: 0.75723 ± 0.00295
```

当前最强非 TTA 训练模型：

```text
实验: P_smp_resnet34_cls2w125_os15
模型: SMP-Unet
encoder: resnet34
encoder_weights: imagenet
class weights: [1, 1, 1.25, 1]
class2 oversampling: 1.5
best val_dice: 0.76493 @ E94
best val_jaccard: 0.67687
per-class Dice @best: class0=0.8552, class1=0.8065, class2=0.6291, class3=0.8382
final val_dice: 0.76251
tail10 val_dice: 0.76254 ± 0.00129
```

判断：

```text
当前最高峰值: P_smp_resnet34_cls2w125_tta, val_dice = 0.76646
当前最强训练模型: P_smp_resnet34_cls2w125_os15, val_dice = 0.76493
当前主线: SMP-Unet ResNet34 + class2 oversampling + TTA / fine-tuning
旧 MultiResUNet 主线: 暂停
```

## 关键结果演化

```text
A_plain_baseline                         best val_dice ≈ 0.6223
B_combined_loss                          best val_dice ≈ 0.6327
B_combined_mild_aug_150                  best val_dice = 0.66530
C_l04_ramp35_scale075                    best val_dice = 0.68139
C_scale075_l04_cls2w15                   best val_dice = 0.68358
C_scale075_l04_cls2w125                  best val_dice = 0.68898
P_smp_resnet34_cls2w125                  best val_dice = 0.75171
P_smp_resnet34_cls2w125_os15             best val_dice = 0.76493
P_smp_resnet34_cls2w125_tta              best val_dice = 0.76646
```

主要增益来源：

```text
combined loss + mild aug                 +约 0.033
scale-factor 0.5 -> 0.75                +约 0.014
class2 weight 1.25                       +约 0.0076 vs 无 class weight
SMP-Unet ResNet34 替代 MultiResUNet       +约 0.063
SMP-Unet + class2 oversampling 1.5        +约 0.013
SMP-Unet + TTA                            +约 0.015 vs SMP 基线
```

## 早期阶段摘要

### 阶段 1：基础 A/B

```text
A_plain_baseline     ≈ 0.6223
B_combined_loss      ≈ 0.6327
B_mild_aug           ≈ 0.6264
B_strong_aug         ≈ 0.5961
B_focal_loss         ≈ 0.5239
```

结论：

1. `combined_loss` 有效。
2. `mild_aug` 稳定但峰值有限。
3. `strong_aug` 从头训练太难。
4. `focal_loss` 单独路线不成立。

### 阶段 2：combined loss + mild aug

```text
B_combined_mild_aug_150
best val_dice = 0.66530 @ E46
```

结论：两个有效因素叠加后明显提升，但 E46 后平台。

### 阶段 3：课程增强

```text
full moderate, max_level=1.0      ≈ 0.658
level≈0.35-0.4                   最有效
C_curriculum_mild_to_moderate_l06 = 0.66726
```

结论：增强不是越强越好，固定 `cosine curriculum + max_level=0.4` 成为后续默认。

### 阶段 4：输入分辨率

```text
C_l04_ramp35_scale075
best val_dice = 0.68139 @ E67
per-class Dice = 0.7685 / 0.7298 / 0.4259 / 0.7670
```

结论：

1. `scale-factor=0.75` 是旧架构最大单次增益。
2. `scale=1.0` 多次不稳定。
3. `batch-size=24` 明显变差。

### 阶段 5：class2 loss weight

```text
C_scale075_l04_cls2w15              0.68358
C_scale075_l04_cls2w125             0.68898
C_scale075_l04_cls011_cls2w15       0.68639
C_scale075_l04_cls2w20              0.65974
```

结论：

1. 旧 MultiResUNet 下 `class2 weight=1.25` 峰值最高。
2. `class2 weight=1.5` 更稳但会牺牲 class0/class1。
3. `class2 weight=2.0` 过强。
4. 单纯继续调 loss 权重已接近平台。

## 阶段 6：SMP-Unet / 预训练 encoder

本轮引入 `segmentation_models_pytorch` 的 Unet：

```bash
--model-architecture smp_unet
--encoder-name resnet34
--encoder-weights imagenet
```

结果：

```text
P_smp_resnet34_cls2w125          best=0.75171 @E80
P_smp_resnet34_random_cls2w125   best=0.74743 @E100
P_smp_effb3_cls2w125             best=0.75174 @E85
```

关键判断：

1. SMP-Unet 直接把上限从 `0.689` 推到 `0.75+`，是当前最大突破。
2. `resnet34 none` 随机初始化也达到 `0.74743`，说明收益主要来自 SMP-Unet 架构，而不是只来自 ImageNet 预训练。
3. ImageNet 权重仍有收益，但不是决定性因素。
4. EfficientNet-B3 没有超过 ResNet34，暂不作为主线。

## 阶段 7：SMP-Unet + class2 oversampling / TTA

结果：

```text
P_smp_resnet34_cls2w125             best=0.75171 @E80
P_smp_resnet34_cls2w125_os15        best=0.76493 @E94
P_smp_resnet34_cls2w125_tta         best=0.76646 @E40
```

per-class 对比：

```text
P_smp_resnet34_cls2w125
class Dice = 0.8279 / 0.7908 / 0.6015 / 0.8486

P_smp_resnet34_cls2w125_os15
class Dice = 0.8552 / 0.8065 / 0.6291 / 0.8382

P_smp_resnet34_cls2w125_tta
class Dice = 0.8448 / 0.8007 / 0.6251 / 0.8456
```

结论：

1. `class2 oversampling=1.5` 明确有效，非 TTA best 从 `0.75171` 到 `0.76493`。
2. class2 Dice 从 `0.6015` 提升到 `0.6291`，且整体 Dice 同步提升。
3. TTA 是有效的推理增强，当前最高 `0.76646`。
4. `os15` 的 tail10 非常稳：`0.76254 ± 0.00129`，说明它是真正强训练模型。
5. MultiResUNet + oversampling 明显失败：

```text
C_scale075_l04_cls2w125_os15       best≈0.6342
C_scale075_l04_cls2w11_os20        best≈0.6426
```

因此 oversampling 后续只在 SMP-Unet 路线上继续。

## 关于 epochs=200 的观察

用户观察：

```text
temp.sh 第 6 条 P_smp_resnet34_cls2w125
100 轮时未明显到达更优
单纯改成 200 轮后，50 多轮达到最优并早停
```

解释：

当前代码原本使用：

```python
CosineAnnealingLR(optimizer, T_max=epochs)
```

因此把：

```bash
--epochs 100
```

改成：

```bash
--epochs 200
```

不仅改变了最大训练轮数，也改变了学习率曲线。`T_max=200` 会让学习率下降更慢，这不是“单纯多跑”，而是换了一条 LR schedule。

结论：

```text
不建议靠简单拉长 epochs 来判断是否能继续提升。
长训时应固定 cosine 周期，例如：
--epochs 140 --lr-cosine-t-max 100
```

## 最新代码能力更新

### 1. Validation TTA

```bash
--val-tta flips
```

四路预测概率平均：

```text
原图
水平翻转
垂直翻转
水平+垂直翻转
```

### 2. class2 oversampling

```bash
--oversample-class-indices 2
--oversample-factor 1.5
--oversample-min-pixels 1
```

只在训练 split 内复制包含 class2 的样本，验证集不动。

### 3. SMP-Unet 接口

```bash
--model-architecture smp_unet
--encoder-name resnet34
--encoder-weights imagenet
```

也支持：

```bash
--encoder-weights none
```

作为随机初始化对照。

### 4. 长训固定 cosine 周期

新增：

```bash
--lr-cosine-t-max 100
```

用途：

```text
允许 --epochs 增加，但不改变原本 100 epoch 的 cosine 学习率节奏。
```

### 5. 减少磁盘占用

新增：

```bash
--checkpoint-interval 0
```

关闭 `model_epoch_N.pth` 周期保存。

推荐所有后续实验默认加：

```bash
--tb-image-interval 0 --tb-num-images 0 --checkpoint-interval 0
```

含义：

1. 不保存 TensorBoard validation image panels。
2. 不保存每 10 轮一次的 epoch checkpoint。
3. 仍保留 `best_model.pth` 和最终 `model.pth`。

## 当前 temp.sh 实验计划

当前 `temp.sh` 已写入 8 条命令，全部关闭 TensorBoard 图片和周期 checkpoint。

```text
1. P_smp_resnet34_cls2w125_os15_tta
   组合两个已验证有效因素：oversampling + TTA

2. P_smp_resnet34_cls2w125_os20
   测试 class2 oversampling 2.0 是否超过 1.5

3. P_smp_resnet34_cls2w10_os15
   有 oversampling 后，测试 class2 loss weight 能否降回 1.0

4. P_smp_resnet34_cls2w15_os15
   测试 class2 weight 1.5 是否能和 os15 叠加

5. P_smp_resnet34_cls2w125_os15_long140_tmax100
   长训到 140，但固定 cosine T_max=100

6. P_smp_resnet34_cls2w125_os15_wd1e3
   增加 weight_decay 到 1e-3，测试更强正则化

7. P_smp_resnet34_cls2w125_os15_aug05
   增强上限从 0.4 提到 0.5，测试 SMP 是否能承受更强增强

8. P_smp_resnet50_cls2w125_os15
   测试更强 ResNet50 encoder
```

运行方式：

```bash
cd ~/zjm/ORO1/ORO/MultiResUNet
bash ../temp.sh
```

如果单卡训练，推荐优先顺序：

```text
1. P_smp_resnet34_cls2w125_os15_tta
2. P_smp_resnet34_cls2w125_os20
3. P_smp_resnet34_cls2w10_os15
4. P_smp_resnet34_cls2w125_os15_long140_tmax100
5. P_smp_resnet50_cls2w125_os15
```

## 当前不建议继续投入的方向

```text
1. 旧 MultiResUNet 主线
2. MultiResUNet + oversampling
3. scale0.5 下继续调增强
4. scale1.0 + 当前训练策略
5. batch-size 24 或更大
6. focal loss 单独路线
7. dice_weight 0.35/0.4 作为主线
8. class2 weight 直接拉到 2.0
9. 简单把 epochs 拉长但不固定 cosine T_max
10. EfficientNet-B3 作为下一轮主线
```

## 后续判断标准

每个新实验必须同时看：

```text
1. best val_dice
2. best val_jaccard
3. final val_dice
4. tail10 mean/std
5. per-class Dice，尤其 class2
6. class0/class1 是否被 class2 优化牺牲
7. best epoch 后是否持续回落
8. 是否使用 TTA
```

注意：

```text
TTA 结果可以作为部署/推理分数。
非 TTA 结果更适合作为训练能力和模型本体能力比较。
```

## 当前决策点

```text
当前最高: P_smp_resnet34_cls2w125_tta, val_dice = 0.76646
当前最强训练模型: P_smp_resnet34_cls2w125_os15, val_dice = 0.76493
当前最重要方向: os15 + TTA 是否能突破 0.77
当前第二方向: os20 / class weight 1.0 / class weight 1.5 的平衡点
当前第三方向: resnet50 是否超过 resnet34
```

下一阶段目标：

```text
短期目标: 非 TTA 超过 0.765
短期目标: TTA 超过 0.77
中期目标: class2 Dice 稳定超过 0.64
若所有 SMP 微调停在 0.765~0.77: 转向样本级错误分析、边界损失、TTA/ensemble
```

## 实验引出关系图

```text
ORO Segmentation Training
├── 00_baseline_ab_test
│   ├── A_plain_baseline -> 0.622
│   ├── B_combined_loss -> 0.633
│   ├── B_mild_aug -> 0.626
│   └── B_focal_loss -> 0.524
│       └── 引出: combined loss + mild augmentation
│
├── 01_combined_mild_aug
│   └── B_combined_mild_aug_150 -> 0.665
│       └── 引出: 课程增强
│
├── 02_curriculum_and_resolution
│   ├── curriculum level≈0.35-0.4 -> 0.667
│   └── scale0.75 -> 0.681
│       └── 引出: class2 瓶颈
│
├── 03_class2_loss_weight
│   ├── cls2w15 -> 0.684
│   ├── cls2w125 -> 0.689
│   └── cls2w20 -> 0.660
│       └── 引出: loss 权重平台，尝试新架构
│
├── 04_smp_unet_switch
│   ├── resnet34 imagenet -> 0.752
│   ├── resnet34 random -> 0.747
│   └── efficientnet-b3 -> 0.752
│       └── 结论: SMP-Unet 架构是大突破
│
├── 05_smp_class2_and_tta
│   ├── resnet34 + os15 -> 0.765
│   ├── resnet34 + TTA -> 0.766
│   └── class2 Dice -> 0.629
│       └── 引出: os15 + TTA / os 强度 / resnet50
│
└── 06_next_runs
    ├── os15 + TTA
    ├── os20
    ├── cls2w10 + os15
    ├── cls2w15 + os15
    ├── long140 + fixed T_max=100
    ├── wd1e-3
    ├── aug level 0.5
    └── resnet50 + os15
```
