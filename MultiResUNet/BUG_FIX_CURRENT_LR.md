# Bug 修复记录 - UnboundLocalError: current_lr

## 🐛 问题描述

训练第一个 epoch 后出现错误：
```
Traceback (most recent call last):
  File "/share/home/zjm/ORO/MultiResUNet/train.py", line 480, in <module>
    main()
  File "/share/home/zjm/ORO/MultiResUNet/train.py", line 385, in main
    history = trainStep(
  File "/share/home/zjm/ORO/MultiResUNet/pytorch/MultiResUNet.py", line 525, in trainStep
    writer.add_scalar('Learning_rate', current_lr, epoch+1)
UnboundLocalError: local variable 'current_lr' referenced before assignment
```

## 🔍 根本原因

在 `pytorch/MultiResUNet.py` 的 `trainStep` 函数中：

```python
# 第 525 行 - 尝试使用 current_lr
if writer is not None:
    writer.add_scalar('Learning_rate', current_lr, epoch+1)  # ❌ 未定义

# 第 530 行 - 才给 current_lr 赋值
current_lr = optimizer.param_groups[0]['lr']  # 赋值太晚
```

**问题**：第一个 epoch 执行到第 525 行时，`current_lr` 还未被初始化。

## ✅ 解决方案

在训练循环开始前初始化 `current_lr`：

```python
# 在 scheduler 初始化后添加
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Initialize current_lr before training loop
current_lr = learning_rate  # ✅ 预先初始化
```

## 📁 修改的文件

- [`pytorch/MultiResUNet.py`](file://e:\project\ORO\MultiResUNet\pytorch\MultiResUNet.py) - 第 460 行附近

## 🧪 验证步骤

修复后可以正常运行：

```bash
# 调试模式测试
bash run_training.sh --epochs 5 --data-limit 20 --verbose --debug

# 正常训练
bash run_training.sh --epochs 150 --data-limit 3500 \
  --batch-size 4 --num-workers 8 --prefetch-factor 4 \
  --gradient-clip 1.0 --save-model --tensorboard
```

## 📊 预期输出

```
Epoch [1/150], Loss: 0.6523
Average Dice Coefficient: 0.1523
Average Jaccard Index: 0.0937
  Current learning rate: 0.000100
  Validation Dice: 0.1523, Jaccard: 0.0937
  ✓ New best model saved! (Dice: 0.1523)
```

✅ TensorBoard 现在可以正常记录学习率曲线了！

---

**修复时间**: 2026-03-09  
**影响范围**: 所有使用 TensorBoard 的训练任务  
**修复状态**: ✅ 已完成
