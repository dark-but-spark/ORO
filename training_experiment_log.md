# MultiResUNet Training Experiment Log

本文档总结截至 2026-06-01 的 MultiResUNet 训练思路、每轮实验动机、结果观察，以及后续训练路径。

## 当前最佳结果

当前最优实验为：

```text
C_l04_ramp35_scale075
scale-factor: 0.75
loss: combined loss, BCE:Dice = 0.7:0.3
augmentation: mild -> moderate, cosine curriculum
max augmentation level: 0.4
ramp: start E40, ramp 35 epochs
best val_dice: 0.68139 @ E67
best val_jaccard: 0.56197
tail10 val_dice: 0.67132 ± 0.00419
```

相比早期最佳：

```text
B_combined_mild_aug_150 best val_dice = 0.66530
```

当前提升约 `+0.0161` Dice，主要来自更高输入分辨率 `scale-factor=0.75`。

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
4. 说明增强强度有效区间不是越强越好，而是存在最佳中间强度。

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
tail10 = 0.67132 ± 0.00419
```

这是目前最优结果，且尾段稳定。

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

## 当前待跑实验设计

已经写入 `temp.sh` 的 8 条方向：

```text
1. C_scale075_l04_cls2w15
2. C_scale075_l04_cls2w20
3. C_scale075_l04_cls2w15_dice04
4. C_scale075_l035_cls2w15
5. C_scale075_l04_ramp50_cls2w15
6. C_scale075_l04_cls2w15_lr15e5
7. C_scale100_mild_lr15e5
8. C_scale100_l02_ramp45_cls2w15_lr15e5
```

### 每条实验的目的

1. `cls2w15`  
   测试轻度 class 2 权重是否提升 class 2，同时不伤害其他类。

2. `cls2w20`  
   测试更强 class 2 权重是否有更大收益，风险是其他类别下降。

3. `cls2w15_dice04`  
   增加 Dice Loss 占比，看是否提升分割重叠质量，尤其是 class 2。

4. `l035_cls2w15`  
   降低增强上限，判断 class 2 权重是否需要更温和的增强环境。

5. `ramp50_cls2w15`  
   放慢增强速度，给 class 2 更多适应时间。

6. `lr15e5`  
   降低学习率，看后期稳定性和 tail 是否改善。

7. `scale100_mild_lr15e5`  
   保守重试 full resolution，不加课程增强，判断 scale1.0 是否本身可用。

8. `scale100_l02_ramp45_cls2w15_lr15e5`  
   full resolution + 极轻课程增强，测试更高分辨率是否需要更弱增强。

## 当前不建议继续投入的方向

```text
1. scale0.5 下继续调增强强度
2. batch-size 24 或更大
3. scale1.0 + level0.4 这类强课程增强
4. 原始严格 adaptive 参数
5. focal loss 单独路线
```

## 当前推荐判断标准

每个新实验不只看 best val Dice，还要看：

```text
1. best val_dice
2. best val_jaccard
3. tail10 mean/std
4. train-val gap
5. class2 val_dice 是否提升
6. 其他 class 是否明显掉
```

如果 class2 提升但整体 Dice 不升，需要检查是否牺牲了 class0/1/3。

如果整体 Dice 小幅提升但 tail 很差，不能直接作为最终方案。

## 目前最合理的下一步

优先跑并分析：

```text
C_scale075_l04_cls2w15
C_scale075_l04_cls2w20
C_scale075_l04_cls2w15_dice04
```

如果这三组中 class2 明显提升，下一轮围绕 class weights 继续微调。

如果 class2 没提升，说明 class2 可能不是 loss 权重问题，而是数据标注、目标形态、分辨率或类别定义问题，需要回到样本级预测图检查。

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
│   │   ├── 观察: class2 Dice≈0.42, 其他类约0.72-0.77
│   │   └── 结论: class2 是当前主要瓶颈
│   │
│   └── class_weighted_loss
│       ├── 代码改动: 增加 --class-weights
│       ├── 作用范围: BCE / Focal / Dice / Combined Loss
│       └── 引出下一批实验:
│           ├── C_scale075_l04_cls2w15
│           ├── C_scale075_l04_cls2w20
│           ├── C_scale075_l04_cls2w15_dice04
│           ├── C_scale075_l035_cls2w15
│           ├── C_scale075_l04_ramp50_cls2w15
│           ├── C_scale075_l04_cls2w15_lr15e5
│           ├── C_scale100_mild_lr15e5
│           └── C_scale100_l02_ramp45_cls2w15_lr15e5
│
└── current_decision_point
    ├── 当前最佳: C_l04_ramp35_scale075, val_dice = 0.68139
    ├── 当前主线: scale0.75 + fixed cosine curriculum
    ├── 当前瓶颈: class2
    ├── 暂停方向:
    │   ├── scale0.5 增强微调
    │   ├── batch24
    │   ├── 原始 strict adaptive
    │   └── scale1.0 + 强课程增强
    └── 下一步:
        ├── 优先分析 class2 weighted loss
        ├── 若 class2 提升且整体不掉, 继续微调 class weights
        └── 若 class2 不提升, 回到样本级预测图检查数据/标注/类别定义
```
