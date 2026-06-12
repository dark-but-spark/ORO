# MultiResUNet Training Experiment Log

本文档总结截至 2026-06-08 的 MultiResUNet 训练思路、每轮实验动机、结果观察，以及后续训练路径。

## 当前最佳结果

当前最优实验为：

```text
C_scale075_l04_cls2w15
scale-factor: 0.75
loss: combined loss, BCE:Dice = 0.7:0.3
augmentation: mild -> moderate, cosine curriculum
max augmentation level: 0.4
ramp: start E40, ramp 35 epochs
class weights: [1, 1, 1.5, 1]
best val_dice: 0.68358 @ E67
best val_jaccard: 0.56341
per-class Dice @best: class0=0.7444, class1=0.7152, class2=0.4729, class3=0.7784
```

相比前一代最佳：

```text
C_l04_ramp35_scale075 best val_dice = 0.68139
```

当前提升约 `+0.00219` Dice。提升幅度不大，但 class 2 从约 `0.4259` 提升到 `0.4729`，说明 class 2 权重方向有效；代价是 class0/class1 有一定下降。

相比早期最佳：

```text
B_combined_mild_aug_150 best val_dice = 0.66530
```

当前累计提升约 `+0.0183` Dice，主要来自 `scale-factor=0.75`，其次来自 class 2 定向加权。

## 阶段 1：基础 A/B 测试

### 实验

最初比较了以下配置：

```text
A_plain_baseline
B_combined_loss
B_focal_loss
B_lr_step
B_mild_aug
B_strong_aug
```

### 结果观察

```text
B_combined_loss      best val_dice ≈ 0.6327
B_mild_aug           best val_dice ≈ 0.6264
A_plain_baseline     best val_dice ≈ 0.6223
B_strong_aug         best val_dice ≈ 0.5961
B_focal_loss         best val_dice ≈ 0.5239
```

### 思考路径

1. `combined_loss` 单独优于 baseline，说明损失设计有帮助。
2. `mild_aug` 单独略低于 combined loss，但泛化更稳。
3. `strong_aug` 训练 Dice 很低，说明从头使用强增强会让模型学不动。
4. `focal_loss` 明显变差，不作为主线。
5. `lr_step` 没解决泛化问题，训练集提升但验证集没有明显提升。

### 引出的下一步

将两个有效因素组合：

```text
combined_loss + mild_aug
```

## 阶段 2：combined loss + mild augmentation

### 实验

```text
B_combined_mild_aug_150
```

### 结果观察

```text
best val_dice = 0.66530 @ E46
```

相比 `B_combined_loss` 的 `0.6327` 有明显提升。

### 思考路径

1. `combined_loss` 提供更好的优化目标。
2. `mild_aug` 控制过拟合。
3. 两者结合后 train/val 更接近，验证 Dice 显著提高。
4. E46 后进入平台，没有明显继续上涨。

### 引出的下一步

既然 mild 已经到平台，而 strong 从头训练太难，尝试课程增强：

```text
mild -> moderate/strong
```

## 阶段 3：固定课程增强 level 扫描

### 实验

围绕 `combined_loss + mild -> moderate`，测试：

```text
level = 1.0
level = 0.2 / 0.4 / 0.6
```

### 结果观察

`level=1.0`：

```text
C_curriculum_mild_to_moderate
best val_dice ≈ 0.6580
```

推满到 moderate 后，表现反而低于 `combined_mild_aug_150`。

`level=0.6`：

```text
C_curriculum_mild_to_moderate_l06
best val_dice = 0.66726 @ E61
level@best ≈ 0.36
```

### 思考路径

1. 完整 moderate 太强，会压低验证上限。
2. 最佳点出现在 level 约 `0.35-0.4`。
3. 后续继续升到 `0.6` 后没有继续提升。
4. 说明增强强度不是越强越好，而是存在最佳中间强度。

### 引出的下一步

设计 adaptive 增强调度：只有模型恢复到上一阶段水平后，才继续提高增强 level。

## 阶段 4：adaptive 增强调度

### 第一次 adaptive

参数较严格：

```text
window = 3
tolerance = 0.002
min_level_epochs = 4
max_level = 0.6
```

结果：

```text
C_adaptive_l06
best val_dice ≈ 0.6557
level 最终只到 0.05
```

### 思考路径

规则太保守，验证集波动导致 level 无法继续上升，模型长期停留在过弱增强下，并出现过拟合。

### 第二次 adaptive loose

放宽参数：

```text
window = 1
tolerance = 0.008
min_level_epochs = 2
max_level = 0.45 / 0.6
```

结果：

```text
C_adaptive_l045_loose  best val_dice = 0.66677
C_adaptive_l06_loose   best val_dice = 0.67628
```

### 思考路径

1. 放宽后 adaptive 能推进到有效 level。
2. 但在 `scale=0.5` 下仍未超过后续的 scale0.75 固定课程增强。
3. adaptive 有帮助，但不是最大提升来源。

### 引出的下一步

继续寻找更有潜力的变量：输入分辨率。

## 阶段 5：提高输入分辨率

### 实验

在当前较优增强策略上，将：

```text
scale-factor 0.5 -> 0.75
```

实验：

```text
C_l04_ramp35_scale075
```

### 结果观察

```text
best val_dice = 0.68139 @ E67
best val_jaccard = 0.56197
tail10 = 0.67132 ± 0.00419
per-class Dice @best = 0.7685 / 0.7298 / 0.4259 / 0.7670
```

### 思考路径

1. 分割任务对边界和细节敏感。
2. `scale=0.5` 可能丢失了 class 2 或边缘区域信息。
3. `scale=0.75` 直接带来约 `+0.014` Dice，说明分辨率是当前主要增益点。

### 引出的下一步

测试：

```text
scale0.75 + adaptive
scale1.0
batch size 24
```

## 阶段 6：scale0.75 后的扩展测试

### 实验

```text
C_scale075_adaptive_l06_loose
C_l04_ramp35_scale100_bs8
C_l04_ramp35_scale075_bs24
```

### 结果观察

```text
C_scale075_adaptive_l06_loose  best val_dice ≈ 0.6726
C_l04_ramp35_scale100_bs8      best val_dice ≈ 0.6417
C_l04_ramp35_scale075_bs24     best val_dice ≈ 0.6149
```

### 思考路径

1. `scale0.75 + fixed cosine level0.4` 仍是最优。
2. `adaptive` 在 scale0.75 下没有超过固定课程增强。
3. `scale1.0` 在当前配置下不稳定，进入增强后验证集崩。
4. `batch24` 明显变差，不应继续加 batch。

### 引出的下一步

停止继续围绕 adaptive、batch、scale1.0 做大规模搜索，转向 class 2 定向优化。

## 阶段 7：class 2 瓶颈分析

### 观察

当前最佳 `C_l04_ramp35_scale075` 的 per-class Dice：

```text
class0 ≈ 0.7685
class1 ≈ 0.7298
class2 ≈ 0.4259
class3 ≈ 0.7670
```

class 2 明显低于其他类别，是整体 Dice 的主要瓶颈。

### 思考路径

继续提高整体 Dice 的关键不再是：

```text
更强增强
更大 batch
更高 scale
```

而是让模型更重视 class 2。

### 代码改动

新增 `--class-weights` 接口，例如：

```bash
--class-weights 1 1 1.5 1
--class-weights 1 1 2.0 1
```

该权重作用于：

```text
BCEWithLogits
Focal Loss
Dice Loss
Combined Loss
```

默认不传时保持旧行为。

## 阶段 8：class-weighted loss 结果

### 实验结果

```text
C_scale075_l04_cls2w15              best val_dice = 0.68358 @ E67
C_scale075_l035_cls2w15             best val_dice = 0.68147 @ E67
C_scale075_l04_cls2w15_dice04       best val_dice = 0.68017 @ E67
C_scale075_l04_ramp50_cls2w15       best val_dice = 0.67671 @ E68
C_scale075_l04_cls2w20              best val_dice = 0.65974 @ E67
C_scale075_l04_cls2w15_lr15e5       best val_dice = 0.65129 @ E61
C_scale100_mild_lr15e5              best val_dice = 0.64465 @ E57
C_scale100_l02_ramp45_cls2w15_lr15e5 best val_dice = 0.64090 @ E57
```

### 关键观察

1. `class2 weight = 1.5` 是唯一明确正收益，整体 Dice 从 `0.68139` 到 `0.68358`。
2. class 2 Dice 从约 `0.4259` 提升到 `0.4729`，说明权重确实让模型更关注 class 2。
3. `class2 weight = 2.0` 明显过强，整体 Dice 掉到 `0.65974`。
4. `dice_weight = 0.4` 能继续推高 class 2 到约 `0.4934`，但整体 Dice 降到 `0.68017`，说明只加 Dice 比例会牺牲其他类。
5. `max_level=0.35` 与 `0.4` 基本接近，但没有超过 `0.4`。
6. `ramp_epochs=50` 变差，说明增强推进变慢不是当前瓶颈。
7. `learning_rate=1.5e-5` 明显变差，说明当前模型需要至少 `2e-5` 的学习率。
8. `scale1.0` 再次失败，当前架构和训练策略下不应继续大规模投入 full resolution。

### 思考路径

class 2 加权方向有效，但收益边际很小，并且会挤压 class0/class1。下一步不应继续粗暴提高 class 2 权重，而应在 `1.25-1.6` 附近细扫，同时尝试轻微补偿 class0/class1，避免 class2 提升换来其他类下降。

## 当前待跑实验设计

已经写入 `temp.sh` 的 6 条聚焦方向：

```text
1. C_scale075_l04_cls2w125
2. C_scale075_l04_cls2w135
3. C_scale075_l04_cls2w16
4. C_scale075_l04_cls011_cls2w15
5. C_scale075_l04_cls2w15_dice035
6. C_scale075_l04_cls2w15_lr25e5
```

### 每条实验的目的

1. `cls2w125`  
   判断 `class2 weight=1.5` 是否已经过强，测试更温和的 `1.25`。

2. `cls2w135`  
   扫描 `1.25` 与当前最佳 `1.5` 之间的中间点。

3. `cls2w16`  
   轻微高于当前最佳 `1.5`，但避开已验证失败的 `2.0`。

4. `cls011_cls2w15`  
   在保持 class2=1.5 的同时补偿 class0/class1，检查能否保住 class2 提升并减少其他类损失。

5. `cls2w15_dice035`  
   测试 `dice_weight=0.35`，位于已知较稳的 `0.3` 和较弱的 `0.4` 之间。

6. `cls2w15_lr25e5`  
   轻微提高学习率到 `2.5e-5`。如果有效，下一轮再考虑 `3e-5`。

## 当前不建议继续投入的方向

```text
1. scale0.5 下继续调增强强度
2. batch-size 24 或更大
3. scale1.0 + 当前课程增强路线
4. 原始严格 adaptive 参数
5. focal loss 单独路线
6. 继续增加 ramp_epochs，例如 ramp50
7. 继续提高 dice_weight 到 0.4 或更高
8. class2 weight 直接拉到 2.0
9. learning rate 降到 1.5e-5
```

## 当前推荐判断标准

每个新实验不只看 best val Dice，还要看：

```text
1. best val_dice
2. best val_jaccard
3. tail10 mean/std
4. train-val gap
5. class2 val_dice 是否提升
6. class0/class1/class3 是否明显下降
```

如果 class2 提升但整体 Dice 不升，需要检查是否牺牲了 class0/1/3。

如果整体 Dice 小幅提升但 tail 很差，不能直接作为最终方案。

## 目前最合理的下一步

优先跑并分析 `temp.sh` 中的 6 条聚焦实验：

```text
C_scale075_l04_cls2w125
C_scale075_l04_cls2w135
C_scale075_l04_cls2w16
C_scale075_l04_cls011_cls2w15
C_scale075_l04_cls2w15_dice035
C_scale075_l04_cls2w15_lr25e5
```

如果其中有实验超过 `0.684`，继续围绕该方向做小范围复验和 seed 稳定性测试。

如果全部停在 `0.681-0.684`，说明当前调参路线接近平台。下一步应转向：

```text
1. 样本级预测图检查 class2 错误来源
2. class2 oversampling 或 patch 采样策略
3. 预训练 encoder / 更强 backbone
4. boundary loss 或边界辅助监督
```

## 实验引出关系图

下面用类似文件目录的形式整理从 baseline 到当前训练计划的演化关系。

```text
MultiResUNet Training
├── 00_baseline_ab_test
│   ├── A_plain_baseline
│   │   └── 观察: train Dice 高, val Dice 较低, 存在过拟合
│   ├── B_combined_loss
│   │   └── 观察: val Dice 优于 baseline
│   │       └── 引出: combined loss 是有效优化方向
│   ├── B_focal_loss
│   │   └── 观察: val Dice 明显下降
│   │       └── 结论: 暂停 focal loss 单独路线
│   ├── B_mild_aug
│   │   └── 观察: 泛化更稳, 但峰值略低于 combined loss
│   │       └── 引出: mild augmentation 有正则化价值
│   ├── B_strong_aug
│   │   └── 观察: train Dice 很低, 模型学不动
│   │       └── 结论: strong aug 不适合从头训练
│   └── B_lr_step
│       └── 观察: train 提升, val 不同步提升
│           └── 结论: LR step 不能解决主要泛化瓶颈
│
├── 01_combine_effective_factors
│   └── B_combined_mild_aug_150
│       ├── 来源: B_combined_loss + B_mild_aug
│       ├── 结果: best val_dice = 0.66530
│       └── 观察: E46 后进入平台
│           └── 引出: mild aug 已接近上限, 尝试课程增强
│
├── 02_curriculum_augmentation
│   ├── C_curriculum_mild_to_moderate
│   │   ├── 设置: mild -> full moderate, max_level = 1.0
│   │   ├── 结果: best val_dice ≈ 0.658
│   │   └── 观察: full moderate 太强, 压低峰值
│   │       └── 引出: 不应直接推满增强
│   │
│   ├── C_curriculum_mild_to_moderate_l02/l04/l06
│   │   ├── 设置: 扫描 max_level = 0.2 / 0.4 / 0.6
│   │   ├── 结果: l06 在 level≈0.36 达到 0.66726
│   │   └── 观察: 最佳区间约 level = 0.35-0.4
│   │       └── 引出: 需要让增强速度匹配模型学习速度
│   │
│   └── adaptive_curriculum
│       ├── C_adaptive_l06
│       │   ├── 设置: 严格 adaptive
│       │   ├── 结果: 只升到 level=0.05, best≈0.6557
│       │   └── 观察: 规则太保守, 增强推进不足
│       │       └── 引出: 放宽 adaptive 判断条件
│       │
│       ├── C_adaptive_l045_loose
│       │   ├── 设置: window=1, tolerance=0.008, min_level_epochs=2
│       │   ├── 结果: best≈0.66677
│       │   └── 观察: 可以推进到有效 level, 但未明显突破
│       │
│       └── C_adaptive_l06_loose
│           ├── 设置: max_level=0.6, loose adaptive
│           ├── 结果: best≈0.67628 at scale0.5
│           └── 观察: 有提升, 但仍低于后续 scale0.75 固定课程
│
├── 03_resolution_scaling
│   ├── C_l04_ramp35_scale075
│   │   ├── 来源: 02 中最佳增强区间 level≈0.4
│   │   ├── 设置: scale-factor = 0.75, max_level = 0.4
│   │   ├── 结果: best val_dice = 0.68139
│   │   └── 观察: 分辨率是当前最大增益来源
│   │       └── 引出: 当前主线切换为 scale0.75
│   │
│   ├── C_scale075_adaptive_l06_loose
│   │   ├── 来源: 想验证 scale0.75 下 adaptive 是否继续提升
│   │   ├── 结果: best≈0.67261
│   │   └── 观察: adaptive 不如固定 l04 ramp35
│   │       └── 结论: scale0.75 下固定课程优先
│   │
│   ├── C_l04_ramp35_scale100_bs8
│   │   ├── 来源: 想测试更高分辨率是否继续提升
│   │   ├── 结果: best≈0.64169, 后期崩
│   │   └── 观察: scale1.0 + 当前增强不适配
│   │       └── 引出: 如重试 scale1.0, 必须更保守
│   │
│   └── C_l04_ramp35_scale075_bs24
│       ├── 来源: 资源充足, 测试更大 batch
│       ├── 结果: best≈0.61486
│       └── 观察: batch24 明显变差
│           └── 结论: 继续使用 batch16
│
├── 04_class_bottleneck
│   ├── per_class_analysis
│   │   ├── 来源: TensorBoard/logging 增加 per-class Dice
│   │   ├── 观察: class2 Dice≈0.426, 其他类约0.72-0.77
│   │   └── 结论: class2 是当前主要瓶颈
│   │
│   └── class_weighted_loss
│       ├── 代码改动: 增加 --class-weights
│       ├── 作用范围: BCE / Focal / Dice / Combined Loss
│       ├── C_scale075_l04_cls2w15
│       │   ├── 结果: best val_dice = 0.68358
│       │   ├── class2: 0.4259 -> 0.4729
│       │   └── 观察: 当前最佳, 但 class0/class1 有下降
│       ├── C_scale075_l04_cls2w20
│       │   ├── 结果: best val_dice = 0.65974
│       │   └── 观察: class2 权重过强, 明显反噬
│       ├── C_scale075_l04_cls2w15_dice04
│       │   ├── 结果: best val_dice = 0.68017
│       │   └── 观察: class2 更高但整体变差, dice_weight=0.4 不作为主线
│       ├── C_scale075_l035_cls2w15
│       │   ├── 结果: best val_dice = 0.68147
│       │   └── 观察: level0.35 与 0.4 接近, 但没超过当前最佳
│       ├── C_scale075_l04_ramp50_cls2w15
│       │   ├── 结果: best val_dice = 0.67671
│       │   └── 观察: ramp 变慢无益
│       ├── C_scale075_l04_cls2w15_lr15e5
│       │   ├── 结果: best val_dice = 0.65129
│       │   └── 观察: lr 降低明显有害
│       └── scale100_retry
│           ├── C_scale100_mild_lr15e5: best≈0.64465
│           ├── C_scale100_l02_ramp45_cls2w15_lr15e5: best≈0.64090
│           └── 结论: 当前不继续投入 scale1.0
│
└── current_decision_point
    ├── 当前最佳: C_scale075_l04_cls2w15, val_dice = 0.68358
    ├── 当前主线: scale0.75 + fixed cosine curriculum + class2 weight 细调
    ├── 当前瓶颈: class2 提升会牺牲 class0/class1, 需要找权重平衡点
    ├── 暂停方向:
    │   ├── scale0.5 增强微调
    │   ├── batch24
    │   ├── 原始 strict adaptive
    │   ├── scale1.0 + 当前训练策略
    │   ├── class2 weight=2.0
    │   ├── dice_weight=0.4
    │   └── lr=1.5e-5
    └── 下一步:
        ├── 运行 6 条 class-weight / loss / lr 聚焦实验
        ├── 若突破 0.684, 做 seed 复验
        ├── 若仍平台, 做 class2 样本级错误分析
        └── 后续考虑 class2 oversampling、预训练 encoder、boundary loss
```

## 最新 temp.sh 训练命令摘要

完整命令已写入 `temp.sh`。当前这批实验应从服务器的 `MultiResUNet` 目录运行：

```bash
cd ~/zjm/ORO1/ORO/MultiResUNet
bash ../temp.sh
```

这批命令的共同基线为：

```text
scale-factor = 0.75
batch-size = 16
learning-rate = 2e-5
loss = combined loss, BCE:Dice = 0.7:0.3
augmentation = mild -> moderate
curriculum = cosine
start epoch = 40
ramp epochs = 35
max aug level = 0.4
early-stopping-min-epochs = 90
early-stopping-patience = 25
```

唯一变量分别是 class2 权重、class0/class1 补偿、Dice 比例和学习率。
