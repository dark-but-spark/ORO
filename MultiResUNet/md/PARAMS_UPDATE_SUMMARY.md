# MultiResUNet 训练参数更新总结

## 📋 更新内容

已为 `train.py` 添加了完整的命令行参数支持，便于调试和灵活配置训练过程。

## ✅ 新增功能

### 1. **数据加载参数**
- `--data-limit`: 控制训练样本数量（默认：100）
- `--validation-split`: 验证集比例（默认：0.2）

### 2. **模型配置参数**
- `--input-channels`: 输入图像通道数（默认：3）
- `--output-channels`: 输出分割通道数（默认：4）

### 3. **训练超参数**
- `--epochs`: 训练轮次（默认：50）
- `--batch-size`: 批次大小（默认：2）
- `--learning-rate`: 初始学习率（默认：1e-4）

### 4. **优化参数**
- `--gradient-clip`: 梯度裁剪阈值（默认：1.0）
- `--weight-decay`: 权重衰减（默认：0）

### 5. **日志和保存**
- `--verbose`: 详细日志输出
- `--save-model`: 保存模型检查点
- `--save-dir`: 模型保存目录（默认：models）

### 6. **调试选项**
- `--debug`: 调试模式
- `--check-data`: 训练前数据验证
- `--device`: 设备选择（cuda/cpu）

## 🚀 快速使用

### 查看帮助
```bash
python train.py --help
```

### 快速测试
```bash
python train.py --data-limit 10 --epochs 5
```

### 调试模式
```bash
python train.py --data-limit 10 --epochs 5 --debug --verbose --check-data
```

### 保存最佳模型
```bash
python train.py --data-limit 100 --epochs 50 --batch-size 4 --save-model
```

## 📊 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-limit` | int | 100 | 训练样本数量 |
| `--validation-split` | float | 0.2 | 验证集比例 |
| `--input-channels` | int | 3 | 输入通道数 |
| `--output-channels` | int | 4 | 输出通道数 |
| `--epochs` | int | 50 | 训练轮次 |
| `--batch-size` | int | 2 | 批次大小 |
| `--learning-rate` | float | 1e-4 | 学习率 |
| `--gradient-clip` | float | 1.0 | 梯度裁剪 |
| `--weight-decay` | float | 0 | 权重衰减 |
| `--verbose` | flag | False | 详细日志 |
| `--save-model` | flag | False | 保存模型 |
| `--save-dir` | str | models | 保存目录 |
| `--debug` | flag | False | 调试模式 |
| `--check-data` | flag | False | 数据验证 |
| `--device` | str | cuda | 设备选择 |

## 🎯 使用场景

### 1. 快速验证代码
```bash
python train.py --data-limit 5 --epochs 3
```

### 2. 数据问题排查
```bash
python train.py --data-limit 10 --check-data --debug
```

### 3. 超参数调优
```bash
python train.py --data-limit 50 --epochs 20 --learning-rate 1e-3 --batch-size 4
```

### 4. 正式训练
```bash
python train.py --data-limit 500 --epochs 100 --batch-size 8 --save-model
```

### 5. 显存不足时
```bash
python train.py --batch-size 1 --data-limit 50
```

### 6. 训练不稳定时
```bash
python train.py --gradient-clip 0.5 --learning-rate 5e-5 --debug
```

## 📁 相关文件

- `train.py` - 主训练脚本（已更新）
- `pytorch/MultiResUNet.py` - 模型定义和训练函数（已更新）
- `TRAINING_ARGS_GUIDE.md` - 详细使用指南
- `train_examples.sh` - 命令示例脚本

## 🔄 主要改进

### 之前
- 硬编码配置，难以调整
- 缺乏调试选项
- 无法灵活控制训练过程

### 现在
- ✅ 所有关键参数可通过命令行调整
- ✅ 支持调试模式和数据验证
- ✅ 可保存训练过程中的最佳模型
- ✅ 支持不同的训练配置快速切换
- ✅ 详细的日志输出便于问题定位

## 💡 最佳实践

1. **初次运行**：先用小数据集验证
   ```bash
   python train.py --data-limit 10 --epochs 5
   ```

2. **数据验证**：确保数据正确
   ```bash
   python train.py --check-data --debug --data-limit 10
   ```

3. **超参数搜索**：从小规模开始
   ```bash
   python train.py --data-limit 50 --epochs 20 --learning-rate 1e-3
   ```

4. **正式训练**：保存最佳模型
   ```bash
   python train.py --data-limit 500 --epochs 100 --save-model
   ```

5. **问题排查**：使用调试模式
   ```bash
   python train.py --debug --verbose --check-data
   ```

## 📝 示例输出

运行以下命令：
```bash
python train.py --data-limit 20 --epochs 10 --debug --save-model
```

输出示例：
```
============================================================
MultiResUNet Training Configuration
============================================================
Data Limit: 20 samples
Validation Split: 20.0%
Input Channels: 3
Output Channels: 4
Epochs: 10
Batch Size: 2
Learning Rate: 0.0001
Gradient Clipping: 1.0
Weight Decay: 0.0
Device: cuda
Debug Mode: True
Data Validation: False
============================================================
Using device: cuda

Data Validation:
  Original Y shape: (20, 640, 640, 4)
  Y sample unique values: [0. 1.]
  Y value range: [0.0000, 1.0000]
  Y positive pixel ratio: 0.0193

Loading data...
...

Epoch [1/10], Loss: 0.6523
Average Dice Coefficient: 0.1523
Average Jaccard Index: 0.0845
  Current learning rate: 0.000100
  Validation Dice: 0.1523, Jaccard: 0.0845
  ✓ New best model saved! (Dice: 0.1523)
...
```

## ⚠️ 注意事项

1. **显存限制**：batch size 过大会导致显存不足
2. **学习率设置**：过高的学习率可能导致训练不稳定
3. **梯度裁剪**：如果训练崩溃，尝试降低 `--gradient-clip`
4. **数据量**：`--data-limit` 太小会导致模型无法有效学习

## 📖 相关文档

- `TRAINING_GUIDE.md` - 完整训练指南
- `TRAINING_ARGS_GUIDE.md` - 命令行参数详细文档
- `README.md` - 项目说明

## 🎉 总结

现在您可以通过简单的命令行参数调整训练配置，无需修改代码！这大大提高了调试效率和实验灵活性。

**推荐起始命令**：
```bash
python train.py --data-limit 10 --epochs 5 --debug --verbose
```

这将帮助您快速了解训练流程，然后逐步调整参数进行正式训练。
