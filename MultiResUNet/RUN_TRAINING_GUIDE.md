# MultiResUNet 训练运行脚本使用指南

## 📋 概述

提供了两个训练运行脚本，支持日志记录、模型保存和结果自动打包：

- **`run_training.sh`** - Linux/Mac 版本
- **`run_training.bat`** - Windows 版本

## 🚀 快速开始

### Windows 用户

```bash
# 基本使用（默认配置）
run_training.bat

# 自定义配置
run_training.bat --epochs 100 --batch-size 4 --data-limit 200
```

### Linux/Mac 用户

```bash
# 基本使用（默认配置）
bash run_training.sh

# 自定义配置
bash run_training.sh --epochs 100 --batch-size 4 --data-limit 200
```

## 📊 功能特性

### ✅ 自动日志记录
- 所有训练输出自动保存到带时间戳的日志文件
- 同时显示在终端和日志文件中
- 包含完整的训练配置和结果

### ✅ 模型保存
- 自动保存训练过程中的最佳模型
- 保存模型权重和优化器状态
- 支持自定义保存目录

### ✅ 结果打包
- 训练完成后自动创建压缩备份
- 包含日志、模型、训练历史等信息
- 自动清理旧备份（保留最近 10 个）

### ✅ 训练历史
- 保存完整的训练指标历史
- 记录 loss、Dice、Jaccard 等关键指标
- 以 NumPy 格式存储，便于后续分析

### ✅ 配置管理
- 支持命令行参数灵活配置
- 提供 manifest 文件记录所有配置信息
- 便于实验复现和对比

## 🎯 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 50 | 训练轮次 |
| `--batch-size` | int | 2 | 批次大小 |
| `--learning-rate` | float | 1e-4 | 学习率 |
| `--data-limit` | int | 100 | 训练样本数量 |
| `--validation-split` | float | 0.2 | 验证集比例 |
| `--input-channels` | int | 3 | 输入通道数 |
| `--output-channels` | int | 4 | 输出通道数 |
| `--gradient-clip` | float | 1.0 | 梯度裁剪阈值 |
| `--device` | str | cuda | 设备选择（cuda/cpu） |
| `--help` | - | - | 显示帮助信息 |

## 📁 输出文件结构

运行脚本后，会生成以下文件结构：

```
runs/
├── logs/
│   └── training_20260303_114500.log      # 训练日志
├── models/
│   ├── best_model_checkpoint.pth         # 最佳模型
│   └── model_architecture.pth            # 模型结构
├── histories/
│   └── history_20260303_114500.npy       # 训练历史
├── manifest_20260303_114500.txt          # 运行配置清单
└── backup_20260303_114500.tar.gz         # 完整备份
```

## 💡 使用示例

### 1. 快速测试
```bash
# Windows
run_training.bat --epochs 5 --data-limit 10

# Linux
bash run_training.sh --epochs 5 --data-limit 10
```

### 2. 调试模式
```bash
# Windows
run_training.bat --epochs 10 --data-limit 20 --batch-size 1 --verbose

# Linux
bash run_training.sh --epochs 10 --data-limit 20 --batch-size 1 --verbose
```

### 3. 完整训练
```bash
# Windows
run_training.bat --epochs 100 --data-limit 500 --batch-size 8 --save-model

# Linux
bash run_training.sh --epochs 100 --data-limit 500 --batch-size 8 --save-model
```

### 4. 学习率实验
```bash
# 测试高学习率
run_training.bat --epochs 50 --data-limit 200 --learning-rate 1e-3

# 测试低学习率
run_training.bat --epochs 50 --data-limit 200 --learning-rate 5e-5
```

### 5. 多通道分割
```bash
# 4 通道分割任务
run_training.bat --output-channels 4 --input-channels 3

# 单通道二值分割
run_training.bat --output-channels 1 --input-channels 1
```

## 📖 日志文件内容示例

```
========================================
MultiResUNet Training Log
========================================
Timestamp: 20260303_114500
Project Directory: e:\project\ORO\MultiResUNet

Training Configuration:
  Epochs: 50
  Batch Size: 2
  Learning Rate: 0.0001
  Data Limit: 100
  Validation Split: 0.2
  Input Channels: 3
  Output Channels: 4
  Gradient Clip: 1.0
  Device: cuda

Starting training...
========================================
============================================================
MultiResUNet Training Configuration
============================================================
...
Epoch [1/50], Loss: 0.6523
Average Dice Coefficient: 0.1523
Average Jaccard Index: 0.0845
  Current learning rate: 0.000100
  Validation Dice: 0.1523, Jaccard: 0.0845
  ✓ New best model saved! (Dice: 0.1523)
...
========================================
Training completed successfully!
========================================
```

## 📦 备份文件内容

备份压缩包（`.tar.gz`）包含：

1. **训练日志** - 完整的训练输出记录
2. **模型文件** - 最佳模型的权重和结构
3. **训练历史** - 所有评估指标的 NumPy 数组
4. **Manifest 文件** - 运行配置和文件清单

### 解压备份

```bash
# Linux/Mac
tar -xzf backup_20260303_114500.tar.gz -C /your/destination/

# Windows (使用 tar 命令，Windows 10+ 支持)
tar -xzf backup_20260303_114500.tar.gz -C your_destination

# Windows (使用 7-Zip 或其他工具)
# 右键点击 .tar.gz 文件，选择解压
```

## 🔍 查看训练历史

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载训练历史
history = np.load('runs/histories/history_20260303_114500.npy', allow_pickle=True).item()

# 绘制损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_dice'], label='Dice')
plt.plot(history['val_jaccard'], label='Jaccard')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Validation Metrics')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()
```

## 🧹 自动清理机制

脚本会自动管理备份文件：

- **保留策略**：自动保留最近 10 个备份
- **清理时机**：每次运行新训练后
- **清理对象**：超过数量的旧备份文件

如需手动清理：

```bash
# Windows
del runs\backup_*.tar.gz

# Linux
rm runs/backup_*.tar.gz
```

## ⚙️ 环境变量配置（可选）

可以通过环境变量设置默认参数：

### Windows (PowerShell)
```powershell
$env:EPOCHS=100
$env:BATCH_SIZE=4
$env:DATA_LIMIT=500
.\run_training.bat
```

### Linux/Mac
```bash
export EPOCHS=100
export BATCH_SIZE=4
export DATA_LIMIT=500
bash run_training.sh
```

## 📊 实验对比

使用 manifest 文件可以轻松对比不同实验：

```bash
# 查看所有运行的配置
cat runs/manifest_*.txt

# 对比特定两次运行
diff runs/manifest_20260303_114500.txt runs/manifest_20260303_120000.txt
```

## 🎯 最佳实践

1. **首次运行**：先用小数据集测试
   ```bash
   run_training.bat --epochs 5 --data-limit 10
   ```

2. **调试数据**：验证数据加载是否正确
   ```bash
   run_training.bat --data-limit 10 --epochs 5 --debug --check-data
   ```

3. **正式训练**：保存完整结果
   ```bash
   run_training.bat --epochs 100 --data-limit 500 --save-model
   ```

4. **结果分析**：使用保存的历史文件绘制曲线
   ```python
   history = np.load('runs/histories/history_TIMESTAMP.npy', allow_pickle=True)
   ```

5. **备份管理**：定期备份重要的 runs 目录
   ```bash
   # 复制到安全位置
   xcopy /E /I runs D:\Backup\MultiResUNet\
   ```

## ⚠️ 注意事项

### Windows 用户
1. **tar 命令**：Windows 10+ 内置支持，旧版本需要安装 7-Zip
2. **路径长度**：确保路径不超过 Windows 最大长度限制
3. **权限问题**：以管理员身份运行可能避免某些权限问题

### Linux/Mac 用户
1. **执行权限**：首次运行前需要添加执行权限
   ```bash
   chmod +x run_training.sh
   ```
2. **Python 环境**：确保已激活正确的 Python 环境

## 🐛 故障排查

### 问题 1: 训练失败
**查看日志**：
```bash
# Windows
type runs\logs\training_TIMESTAMP.log

# Linux
cat runs/logs/training_TIMESTAMP.log
```

### 问题 2: 备份创建失败
**检查文件**：
```bash
# Windows
dir runs\logs
dir runs\models
dir runs\histories

# Linux
ls -la runs/logs/
ls -la runs/models/
ls -la runs/histories/
```

### 问题 3: 显存不足
**减小 batch size**：
```bash
run_training.bat --batch-size 1 --data-limit 50
```

## 📚 相关文档

- [TRAINING_ARGS_GUIDE.md](TRAINING_ARGS_GUIDE.md) - 命令行参数详细说明
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 完整训练指南
- [PARAMS_UPDATE_SUMMARY.md](PARAMS_UPDATE_SUMMARY.md) - 参数更新总结

## 🎉 总结

使用 `run_training.bat` 或 `run_training.sh` 脚本，您可以：

✅ **自动化训练流程** - 一键启动训练并保存所有结果  
✅ **完整日志记录** - 所有输出自动保存到带时间戳的文件  
✅ **结果备份** - 自动创建包含所有重要文件的压缩包  
✅ **实验管理** - 通过 manifest 文件轻松管理和对比实验  
✅ **灵活配置** - 通过命令行参数快速调整训练设置  

**推荐起始命令**：
```bash
# Windows
run_training.bat --epochs 10 --data-limit 20 --debug

# Linux
bash run_training.sh --epochs 10 --data-limit 20 --debug
```
