# MultiResUNet / SMP-Unet Training Experiment Log

本文档总结截至 2026-07-24 的训练路径、最新运行结果、代码能力更新，以及下一批训练计划。

## 2026-07-24 最新运行结论

本轮最新数据来自：

```text
runsTemp/runsABCtest/logs
experiment_analysis.md
```

当前最高结果已经从 `0.76851` 推进到：

```text
实验: P_smp_resnet34_cls2w125_os15_tta_long140_tmax100_20260724_161215
模型: SMP-Unet
encoder: resnet34
encoder_weights: imagenet
scale-factor: 0.75
class weights: [1, 1, 1.25, 1]
class2 oversampling: 1.5
loss: BCE:Dice = 7:3
augmentation: cosine curriculum, max level=0.4
epochs: 140
lr scheduler: cosine
lr_cosine_t_max: 100
val_tta: flips
best val_dice: 0.77262 @ E114
best val_jaccard: 0.68591
best train_dice: 0.87948
train-val gap @best: +0.10686
final val_dice: 0.76122
completed epochs: 140
stopped early: false
```

### 本轮排序

```text
1. tta_long140_tmax100     0.77262 @E114  TTA + 140ep + T_max=100
2. long140_tmax100         0.76851 @E98   140ep + T_max=100
3. tta_20260719            0.76798 @E91   TTA
4. plateau                 0.76516 @E116  plateau scheduler
5. aug05                   0.76486 @E94   aug max=0.5
6. fullres_bs8             0.76468 @E99   scale=1.0, bs=8
7. wd1e3                   0.76381 @E100  weight_decay=1e-3
8. cls2w10_v1              0.76234 @E92   no class2 upweight
9. lr5e5                   0.76084 @E68   lr=5e-5
10. focaldice73_v1         0.76064 @E64   focal+dice 7:3
11. focaldice73_v2         0.76004 @E115  focal+dice 7:3 repeat
12. focaldice55            0.75952 @E115  focal+dice 5:5
13. focal_fullres_plateau  0.75880 @E81   focal + fullres + plateau
14. resnet50               0.75834 @E108  larger encoder
15. purefocal              0.72062 @E100  pure focal
```

### 新判断

1. `TTA + long140 + T_max=100` 是当前明确最优组合，比无 TTA 长训练提升 `+0.0041`，比 07-19 TTA 提升 `+0.0046`。
2. 140 轮完整跑完且没有早停，说明这个配置不是早停偶然峰值；长周期学习率确实有价值。
3. 当前核心瓶颈仍是过拟合：最优模型 `train-val gap=+0.10686`，没有因为 TTA 消失。
4. `Focal` 路线已经可以停止。纯 Focal 只有 `0.72062`，Focal+Dice 也停在 `0.759-0.761`。
5. `plateau scheduler`、`lr=5e-5`、`resnet50` 都没有超过 cosine + resnet34 主线。
6. `fullres_bs8` 和 `wd1e3` 虽然没有提分，但显著降低 gap，说明下一步应该从“泛化正则化”而不是“更强损失函数”里找突破。

### 下一步优先级

```text
P0: 做 top checkpoints / top models 的概率平均 ensemble，优先验证是否能无训练突破 0.775。
P1: 给 smp_unet 增加真正生效的 decoder/head dropout 接口，再围绕 0.1/0.2 做小范围实验。
P2: 在当前 best 配置上测试 SWA/EMA 或 top-k checkpoint averaging，目标减少 E114 后回落。
P3: fullres + 正则化重跑一条干净对照，不要叠 focal，验证低 gap 是否能转化为 Dice。
P4: 做 class2 和边界错误分析，决定是否值得实现 boundary loss / boundary dice。
```

暂时不建议继续扩大：

```text
1. Focal / pure focal / focal+fullres
2. resnet50
3. aug max 0.5 以上
4. lr 5e-5
5. 单纯增加 epochs 但不改变 T_max、正则化或模型平均
```

## 2026-07-22 最新运行结论

本轮最新数据来自：

```text
runsTemp/runsABCtest/logs
TRAINING_ANALYSIS.md
```

当前最佳已经从 6 月的 `0.76646` 小幅推进到：

```text
实验: P_smp_resnet34_cls2w125_os15_long140_tmax100_20260721_143943
模型: SMP-Unet
encoder: resnet34
encoder_weights: imagenet
scale-factor: 0.75
class weights: [1, 1, 1.25, 1]
class2 oversampling: 1.5
epochs: 140
lr_cosine_t_max: 100
best val_dice: 0.76851 @ E98
best val_jaccard: 0.68045
per-class Dice @best: class0=0.8473, class1=0.8051, class2=0.6113, class3=0.8545
final val_dice: 0.75842
tail10 val_dice: 0.75967 ± 0.00392
```

当前最值得关注的亚军是：

```text
实验: P_smp_resnet34_cls2w125_os15_tta_20260719_221021
best val_dice: 0.76798 @ E91
best val_jaccard: 0.68205
per-class Dice @best: class0=0.8517, class1=0.8121, class2=0.6402, class3=0.8537
final val_dice: 0.76629
tail10 val_dice: 0.76404 ± 0.00200
```

判断：

```text
总 Dice 最强: long140_tmax100, 0.76851
Jaccard / class2 最强: os15_tta_20260719, Jaccard=0.68205, class2=0.6402
当前真实平台: 0.768 左右
当前下一目标: 稳定突破 0.77
```

### 最新实验排序

```text
1. cls2w125_os15_long140_tmax100     0.76851 @E98
2. cls2w125_os15_tta_20260719        0.76798 @E91
3. cls2w125_os15_aug05               0.76486 @E94
4. cls2w125_os15_wd1e3               0.76381 @E100
5. cls2w125_os15_tta_20260720        0.76301 @E44
6. cls2w10_os15_20260720             0.76234 @E92
7. cls2w10_os15_20260721             0.75916 @E108
8. resnet50_cls2w125_os15            0.75834 @E108
```

### 最新消融结论

1. `long140 + lr_cosine_t_max=100` 是当前最高点，但比 `os15_tta` 只高 `0.00053`，收益很小。
2. `TTA` 结果不完全稳定：一条到 `0.76798`，另一条早停在 `0.76301`。但第一条的 final/tail 都很好，不应丢掉。
3. `aug_max=0.5` 没有超过 `0.4`，说明增强上限继续加大不是主线。
4. `weight_decay=1e-3` 降低 train-val gap，但 best Dice 下降到 `0.76381`，适合泛化保守方案，不适合冲最高分。
5. `resnet50` 没有超过 `resnet34`，继续加大 encoder 暂时不是主线。
6. `class2 weight=1.0 + os15` 可达到 `0.76234/0.75916`，说明 class2 权重 `1.25` 仍有必要。

## 参考图实验方向可行性

参考图提出 R1-R8。结合当前代码与已有结果，评估如下：

```text
R1 Focal + Dice (7:3)
可行。当前代码支持 --use-combined-loss --use-focal-loss。
值得试，但优先级中等；旧 MultiResUNet 的 focal 单独路线失败，不代表 SMP+os15 下也失败。

R2 Focal + Dice (5:5)
可行，但风险更高。此前提高 Dice 权重没有收益，5:5 可能牺牲 BCE 的分类校准。

R3 纯 Focal
可行，作为排除实验可以跑一条。预期不高。

R4 LR 5e-5
可行但风险大。旧实验中更高 LR 容易不稳；SMP 可能更能承受，但建议只作为探索。

R5 Plateau LR
可行，且值得试。当前平台明显，ReduceLROnPlateau 比固定 cosine 更符合“卡住后降 LR”的需求。

R6 全分辨率 scale=1.0
可行但成本高。旧架构 full-res 失败，SMP 可能更好；建议 batch=8，先单条验证。

R7 Dropout 0.4
当前对 smp_unet 不可直接生效。--dropout-rate 目前只作用于原 MultiResUNet。
如果要测 SMP dropout，需要额外改代码，在 SMP decoder/head 中显式插入 dropout。

R8 组合拳 Focal + FullRes + Plateau
可行但不适合作为第一批主线。变量太多，若变好/变坏都难归因。
建议在 R1/R5/R6 中至少一条有正信号后再组合。
```

本轮策略：

```text
优先试: R5 Plateau LR, R1 Focal+Dice(7:3), R6 FullRes
谨慎试: R2, R3, R4
暂不直接试: R7，除非先改代码
最后再试: R8 组合拳
```

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

## 2026-07-25 class2 忽略评分与降权小训练

### 背景

最新诊断集中，worst cases 显示 class2 的主要问题不像单纯模型容量不足，更像是标注不全、弱病变标准不一致、以及部分切片不应被强监督。为了判断 class2 是否正在拖累共享特征，本轮先做“只在评分阶段忽略 class2”的复算，再安排一批 class2 降权小训练。

### 新增分析脚本

```text
analyze_ignore_class.py
```

运行方式：

```bash
python analyze_ignore_class.py \
  --diagnostics-dir runsTemp/diagnostics \
  --ignore-classes 2 \
  --output-dir runsTemp/diagnostics_ignore_class_analysis
```

结果：

```text
best_20260724_tta:
  original overall Dice = 0.7735
  macro without class2 = 0.9499
  micro without class2 = 0.8449

best_20260724_no_tta:
  original overall Dice = 0.7663
  macro without class2 = 0.9464
  micro without class2 = 0.8362
```

解释：

```text
macro without class2 容易被空类别和简单类别抬高，不宜作为最终真实分数。
micro without class2 更接近忽略 class2 后的像素级实际表现。
当前可信判断：忽略 class2 后模型约为 0.84 水平，不是 0.95。
```

### 归档

本轮相关代码、报告和训练指令已归档到：

```text
docs/archive/2026-07-25/
```

包括：

```text
class2_ignore_and_next_training.md
analyze_ignore_class.py
temp.sh
ignore_class_report.md
ignore_class_summary.csv
best_20260724_tta_ignore_classes_2_samples.csv
best_20260724_no_tta_ignore_classes_2_samples.csv
diagnostics_analysis.md
experiment_analysis.md
training_experiment_log.md
```

### 下一步训练设计

`temp.sh` 已更新为 class2 降权对照训练。固定当前最强骨架：

```text
smp_unet + resnet34 + imagenet
scale-factor 0.75
combined loss = BCE 0.7 + Dice 0.3
cosine lr, T_max=100
mild -> moderate cosine curriculum, max level 0.4
checkpoint-interval 0
tb-image-interval 0
metric-ignore-classes 2
```

本轮代码已增加训练期忽略 class2 的额外指标，原始 `val_dice` 不变：

```text
Metrics/val_dice_ignore_classes_2
Metrics/val_jaccard_ignore_classes_2
Metrics/val_macro_class_dice_ignore_classes_2
Metrics/val_macro_class_jaccard_ignore_classes_2
```

这些指标会进入 TensorBoard、`training_history.npy` 和 `summary.json`。默认 best model 和 early stopping 仍按包含 class2 的原始 `val_dice`，避免和历史结果混淆。

实验组：

```text
Q_cls2w125_os15_tta_anchor   当前强配置锚点
Q_cls2w10_os15_tta           class2 weight 降到 1.0
Q_cls2w075_os15_tta          class2 weight 降到 0.75
Q_cls2w05_os15_tta           class2 weight 降到 0.5
Q_cls2w075_os12_tta          class2 weight 0.75 + oversample 1.2
Q_cls2w10_noos_tta           class2 weight 1.0 + 去掉 oversample
Q_cls2w075_os15_no_tta       class2 weight 0.75 + no-TTA 验证
```

判断标准：

```text
如果 class2 降权后 overall 或 class0/1/3 上升，而 class2 没有灾难性下降：
  说明 class2 噪声正在拖共享特征，后续应做 class2 置信 mask 或高置信验证集。

如果 class2 降权后 overall 明显下降：
  说明 class2 虽然噪声多，但仍提供关键结构监督，不能简单降权或删除。
```

## 2026-07-27 至 2026-07-29 固定划分、独立 test 与跨源数据检查

### 背景

前期 `P_*` 系列实验主要使用随机/混合验证集，最高验证 Dice 约 0.77。后续人工复核发现 worst cases 中存在较多 GT 漏标、class2 标准不一致、TTA 变差样本，因此继续只调训练参数意义有限。

本阶段目标从“继续堆验证集分数”转为：

```text
1. 固定 train/valid/test，建立可信评估口径。
2. 用独立 test 重新评价历史模型。
3. 检查不同数据源是否存在泄露或域差异。
4. 为后续清洗数据和跨源训练建立基线。
```

### 固定 A 数据集

重新解压并处理 `20260204111923` 数据，保留原始 train/valid/test 划分：

```text
data/20260204111923/train/masks: 2782
data/20260204111923/valid/masks: 794
data/20260204111923/test/masks: 400
```

对应代码已经支持：

```text
--split-mode fixed
--train-img-dir / --train-mask-dir
--val-img-dir / --val-mask-dir
--test-img-dir / --test-mask-dir
```

训练结束后会额外保存 test 结果：

```text
history/test_metrics.json
```

### 固定 A 结果

固定 A 上目前更可信的锚点结果：

```text
U_A_20260204_anchor_scale075_cls2w10_20260728_223444
  model: smp_unet + resnet34 + imagenet
  scale-factor: 0.75
  class weights: 1,1,1,1
  metric-ignore-classes: 2
  best val Dice: 0.5665 @E26
  test Dice: 0.5430
  test Dice ignore class2: 0.6945
  class Dice: [0.6124, 0.7831, 0.4041, 0.7665]

U_A_20260204_fullres_cls2w10_20260729_000602
  scale-factor: 1.0
  best val Dice: 0.5565
  test Dice: 0.5331
  test Dice ignore class2: 0.6735
```

结论：

```text
scale-factor 0.75 优于 full resolution。
class2 仍是主要瓶颈；忽略 class2 后 test Dice 从 0.5430 上升到 0.6945。
旧随机验证 0.77 不能直接和固定 A test 0.54 比较。
```

### R_fixed 对照结果

早期固定/混合 test 459 张上的对照：

```text
R_fixed_smp_resnet34_scale075_cls2w125_20260727_194004
  best val Dice: 0.6011
  test Dice: 0.5681
  test Dice ignore class2: 0.6976

R_fixed_smp_resnet34_scale075_cls2w05_20260727_211702
  best val Dice: 0.6041
  test Dice: 0.5821
  test Dice ignore class2: 0.7153
```

解释：

```text
class2 weight 从 1.25 降到 0.5 后 test 更好，提示 class2 噪声可能正在拖累共享特征。
但该 test 口径与固定 A 的 400 张 test 不完全一致，后续仍应在固定 A 上复验。
```

### B 数据源泄露检查

`385-liver.v1i.yolov8` 是 Roboflow 格式，原始数据中存在 `.rf.` 增强变体。按原始文件名分组后发现 train/valid/test 存在严重同源泄露：

```text
train: 4230 images / 268 original groups
valid: 144 images / 117 original groups
test:   59 images /  56 original groups

valid 与 train 原图组重叠约 86%
test 与 train 原图组重叠约 86%
```

因此原始 B 上的高分不能作为真实泛化能力：

```text
U_B_385liver_anchor_scale075_cls2w10_20260728_231431
  best val Dice: 0.8119
  test Dice: 0.8536

U_B_385liver_fullres_cls2w10_20260729_011138
  best val Dice: 0.8093
  test Dice: 0.8583
```

这些结果更像是“同源增强变体记忆”，不能直接证明模型在独立数据上达到 0.85。

### B_clean 构建

新增/使用脚本：

```text
MultiResUNet/scripts/group_split_roboflow_dataset.py
```

目标：

```text
按 Roboflow 原始图片 stem 分组；
同一原图的增强变体只能进入同一个 split；
每个原图组最多保留 3 个变体；
YOLO 标签同时转为 mask；
输出 leakage-safe 数据集。
```

输出目录：

```text
data/385-liver.groupclean.v1
```

当前清洗后规模：

```text
train: 230 groups / 671 images
valid:  29 groups /  85 images
test:   28 groups /  82 images
```

后续 B 的结论必须优先看 B_clean，而不是原始 B。

### 当前 temp.sh 状态

`temp.sh` 已更新为 B_clean 训练与跨源评估流程：

```text
1. 如果不存在 data/385-liver.groupclean.v1，则自动创建 B_clean。
2. 训练 V_Bclean_anchor_scale075_cls2w10。
3. 评估 Bclean model on Bclean test。
4. 评估 Bclean model on A test。
5. 评估已有 U_A_* models on Bclean test。
```

运行方式：

```bash
cd ~/ORO/MultiResUNet
bash ../temp.sh
```

### 当前阶段结论

```text
1. 项目已经进入“数据质量 + 可信评估”阶段，不再适合只追随机验证集分数。
2. 固定 A 上当前可信 test Dice 约 0.543；忽略 class2 后约 0.695。
3. class0/class1/class3 并不差，class2 是最明显瓶颈。
4. B 原始数据存在泄露，高分不可直接采用。
5. B_clean 可用于测试跨源鲁棒性，但是否适合作为 A 的预训练来源，需要等待 Bclean -> A test 和 A -> Bclean test 结果。
```

### 下一步优先级

```text
P0: 跑完 B_clean 训练与交叉测试，确认 Bclean 是否有可迁移价值。

P1: 固定 A 上做小规模 class2 权重复验：
    cls2w1.0 / cls2w0.75 / cls2w0.5
    主指标看 fixed A test，不再只看 val。

P2: 人工复核 fixed A test worst cases，尤其 class2：
    GT 漏标、炎症标准不一致、单个炎细胞误标、脂肪变/气球样变漏标。

P3: 增加更可靠的评估指标：
    dataset-global Dice、GT-positive-only Dice、per-class macro/micro、lesion-level precision/recall。

P4: 代码层面再尝试 EMA、top-k checkpoint averaging、SMP decoder/head dropout、val-only threshold calibration。

P5: 若 Bclean 跨源表现尚可，再考虑 Bclean 预训练 -> A 微调；
    若跨源表现差，先不要混用 A/B 训练，应优先清洗 A 与建立高置信小验证集。
```

### 2026-07-29 归档

本阶段文档、日志索引和结果汇总已归档到：

```text
docs/archive/2026-07-29/
```

主要文件：

```text
README.md
runsABCtest_logs_inventory.csv
runsABCtest_results_summary.csv
latest_results_focus.csv
root_snapshots/
run_logs/
manual_review/
ignore_class_analysis/
```

说明：

```text
大体积 TensorBoard 事件、模型权重、诊断图片不重复复制，只在 CSV 清单中记录原路径。
```
