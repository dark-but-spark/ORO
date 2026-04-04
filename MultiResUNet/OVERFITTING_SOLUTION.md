# MultiResUNet 过拟合问题解决方案

## 📊 如何识别过拟合

### 典型症状
- ✅ **训练集表现很好**：Dice > 0.9, Loss 持续下降
- ❌ **验证集表现差**：Dice < 0.5, Loss 开始上升或停滞
- ⚠️ **差距逐渐增大**：训练和验证指标之间的gap越来越大

### 通过TensorBoard观察
```bash
tensorboard --logdir runs/logs
```

在TensorBoard中查看：
- `Loss/train` vs `Loss/validation` - 验证loss开始上升是过拟合的明确信号
- `Metrics/val_dice` - 如果 plateau 或下降，说明泛化能力不足
- `Learning_rate` - 确认学习率调度器正常工作

---

## 🎯 过拟合解决方案（按优先级排序）

### 方案1：添加正则化（最简单有效）⭐⭐⭐⭐⭐

#### 方法A：Weight Decay (L2正则化)
```bash
# 轻度正则化（推荐先试这个）
python train.py --weight-decay 1e-5

# 中度正则化（常用配置）
python train.py --weight-decay 1e-4

# 强度正则化（如果过拟合严重）
python train.py --weight-decay 5e-4
```

**原理**：惩罚大的权重值，使模型更平滑

**建议**：从 `1e-5` 开始，逐步增加到 `1e-4`

---

#### 方法B：梯度裁剪（已默认启用）
```bash
# 更强的梯度裁剪
python train.py --gradient-clip 0.5

# 标准配置（默认）
python train.py --gradient-clip 1.0
```

---

### 方案2：调整学习率和训练策略 ⭐⭐⭐⭐

#### 降低学习率
```bash
# 保守学习率（稳定但慢）
python train.py --learning-rate 5e-5

# 非常保守（适合小数据集）
python train.py --learning-rate 1e-5
```

**为什么有效**：较小的学习率让模型在最优解附近小幅震荡，而不是过度拟合训练数据

#### 增加训练轮次 + 早停
```bash
# 更多epoch + 正则化
python train.py --epochs 100 --weight-decay 1e-4 --learning-rate 5e-5
```

**注意**：当前代码使用Cosine Annealing调度器，会自动降低学习率

---

### 方案3：使用更好的损失函数 ⭐⭐⭐⭐

#### 方法A：Combined Loss（BCE + Dice）
```bash
# 平衡两种损失
python train.py --use-combined-loss --bce-weight 0.5 --dice-weight 0.5 --weight-decay 1e-4
```

**优势**：
- BCE关注像素级准确性
- Dice关注整体区域重叠
- 两者结合避免模型偏向某一方面

#### 方法B：Focal Loss（处理类别不平衡）
```bash
# 适用于目标区域稀疏的情况
python train.py --use-focal-loss --focal-alpha 0.25 --focal-gamma 2.0 --weight-decay 1e-4
```

**适用场景**：
- 前景像素占比 < 10%
- 背景占主导的数据集

---

### 方案4：增加训练数据 ⭐⭐⭐⭐⭐

#### 最小要求
```bash
# 至少100个样本
python train.py --data-limit 100

# 推荐200+样本
python train.py --data-limit 200

# 理想500+样本
python train.py --data-limit 500
```

#### 数据增强（强烈推荐！）

⚠️ **当前代码缺少数据增强功能**，建议添加以下增强：

```python
# 在 dataloading.py 中添加数据增强
import albumentations as A

transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50)),
        A.MotionBlur(blur_limit=3),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
    ], p=0.3),
])
```

---

### 方案5：减少模型复杂度 ⭐⭐⭐

MultiResUNet的alpha参数控制网络宽度：

```python
# 在 pytorch/MultiResUNet.py 中修改
class MultiResUnet(nn.Module):
    def __init__(self, input_channels=3, num_classes=4, alpha=1.67):
        # 将alpha从1.67降到1.2-1.4
        self.alpha = alpha  # 尝试 1.2, 1.3, 1.4
```

**效果**：
- alpha=1.67（默认）：较宽的网络，容易过拟合
- alpha=1.4：中等宽度
- alpha=1.2：较窄的网络，泛化更好

---

### 方案6：Batch Size调整 ⭐⭐⭐

```bash
# 更大的batch size通常更稳定
python train.py --batch-size 4 --weight-decay 1e-4

# 如果显存允许
python train.py --batch-size 8 --weight-decay 1e-4
```

**原理**：大batch提供更准确的梯度估计

---

## 🔧 推荐的完整配置组合

### 配置1：轻度过拟合（训练Dice 0.8-0.9，验证Dice 0.5-0.6）

```bash
python train.py \
  --data-limit 200 \
  --epochs 80 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --weight-decay 1e-5 \
  --gradient-clip 1.0 \
  --save-model \
  --tensorboard
```

---

### 配置2：中度过拟合（训练Dice >0.9，验证Dice 0.3-0.5）

```bash
python train.py \
  --data-limit 300 \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --use-combined-loss \
  --bce-weight 0.5 \
  --dice-weight 0.5 \
  --gradient-clip 0.5 \
  --save-model \
  --tensorboard
```

---

### 配置3：严重过拟合（训练Dice >0.95，验证Dice <0.3）

```bash
python train.py \
  --data-limit 500 \
  --epochs 120 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --weight-decay 5e-4 \
  --use-focal-loss \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --gradient-clip 0.5 \
  --save-model \
  --tensorboard
```

---

### 配置4：小数据集防过拟合（<100样本）

```bash
python train.py \
  --data-limit 50 \
  --epochs 150 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --use-combined-loss \
  --gradient-clip 0.5 \
  --save-model \
  --verbose
```

**额外建议**：
- 使用交叉验证
- 人工检查数据质量
- 考虑迁移学习（如果有预训练权重）

---

## 📈 监控和调试技巧

### 1. 实时监控训练过程

```bash
# 在一个终端运行训练
python train.py --tensorboard --verbose

# 在另一个终端启动TensorBoard
tensorboard --logdir runs/logs
```

**关键观察点**：
- Epoch 1-10: 训练和验证loss应该同步下降
- Epoch 10-30: 如果验证loss停止下降或上升 → 过拟合开始
- Epoch 30+: 验证Dice应该趋于稳定

---

### 2. 分析训练历史

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载训练历史
history = np.load('training_history.npy', allow_pickle=True).item()

# 绘制loss曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['dice'], label='Train Dice')
plt.title('Training Dice')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 3, 3)
# 如果有验证集指标，也绘制出来
if 'val_dice' in history:
    plt.plot(history['val_dice'], label='Val Dice')
    plt.plot(history['dice'], label='Train Dice')
    plt.title('Dice Comparison')
    plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

---

### 3. 诊断工具

```bash
# 检查数据质量
python scripts/diagnose_data.py

# 内存诊断
python diagnose_memory.py

# 快速测试（5个样本，5个epoch）
python train.py --data-limit 5 --epochs 5 --debug --check-data
```

---

## 🎓 最佳实践总结

### DOs ✅
1. **从小规模开始**：先用50-100样本测试
2. **逐步增加正则化**：从1e-5到1e-4
3. **使用TensorBoard**：实时监控训练过程
4. **保存多个checkpoint**：比较不同配置的效果
5. **验证数据质量**：确保标签正确
6. **使用Combined Loss**：比单一损失更稳定

### DON'Ts ❌
1. **不要一开始就用大学习率**：容易导致不稳定
2. **不要忽略验证集指标**：只看训练指标会误导
3. **不要在过拟合时继续增加epoch**：应该加强正则化
4. **不要用太小的数据集**：<50样本很难训练好
5. **不要忘记梯度裁剪**：防止梯度爆炸

---

## 🔍 常见问题排查

### Q1: 添加了weight decay但还是过拟合？
**A**: 尝试以下组合：
- 增加weight decay到5e-4
- 降低学习率到1e-5
- 使用Combined Loss
- 增加训练数据

### Q2: 验证Dice波动很大？
**A**: 
- 增加batch size（从2到4或8）
- 减小学习率
- 增加weight decay
- 检查验证集是否太小（至少20%数据）

### Q3: 训练很慢怎么办？
**A**:
- 使用GPU（--device cuda）
- 增加num_workers（--num-workers 4）
- 启用混合精度训练（需要修改代码）
- 减少图像分辨率（--scale --scale-factor 0.5）

### Q4: 如何判断是否还需要更多数据？
**A**:
- 如果训练Dice < 0.7：模型欠拟合，增加模型容量或训练时间
- 如果训练Dice > 0.9但验证Dice < 0.5：过拟合，需要更多数据或更强正则化
- 如果训练和验证Dice都~0.7：可能需要更多数据提升性能

---

## 📚 相关文档

- [训练指南](TRAINING_GUIDE.md) - 完整的训练流程
- [命令行参数指南](TRAINING_ARGS_GUIDE.md) - 所有可用参数
- [OOM修复指南](OOM_FIX_GUIDE.md) - 内存优化技巧
- [TensorBoard日志说明](md/TENSORBOARD_STORAGE_LOGIC.md) - 如何使用TensorBoard

---

**最后更新**: 2026-04-04  
**适用版本**: MultiResUNet v2.0+  
**维护者**: ORO Project Team
