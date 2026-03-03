# MultiResUNet 训练脚本功能更新总结

## 🎉 更新概览

参考 `Pytorch-UNet/run_training.sh` 的设计模式，为 MultiResUNet 添加了完整的训练运行脚本，实现**日志自动记录**和**结果打包备份**功能。

## ✅ 新增文件

### 1. **run_training.bat** (Windows 版本)
- 完整的批处理训练脚本
- 支持所有命令行参数
- 自动日志记录和备份
- 适用于 Windows 10+ 系统

### 2. **run_training.sh** (Linux/Mac 版本)
- Shell 版本的训练脚本
- 与 Windows 版本功能一致
- 适用于 Linux/Mac 系统

### 3. **RUN_TRAINING_GUIDE.md** (使用指南)
- 详细的使用说明文档
- 包含多个实际案例
- 故障排查指南

### 4. **SCRIPT_COMPARISON.md** (对比文档)
- 三种训练方式的对比
- 使用场景推荐
- 最佳实践指南

## 🚀 核心功能

### 1️⃣ **自动日志记录**
```bash
# 所有输出自动保存到带时间戳的日志文件
runs/logs/training_20260303_114500.log
```

**特点**：
- ✅ 终端和文件同步输出（使用 `tee` 命令）
- ✅ 包含完整训练配置
- ✅ 记录所有训练指标
- ✅ 便于后续分析和问题排查

---

### 2️⃣ **结果自动打包**
```bash
# 训练完成后自动创建压缩包
runs/backup_20260303_114500.tar.gz
```

**备份内容**：
- ✅ 训练日志
- ✅ 模型文件
- ✅ 训练历史（NumPy 格式）
- ✅ 配置清单（manifest）

---

### 3️⃣ **配置清单管理**
```bash
# 每次运行生成 manifest 文件
runs/manifest_20260303_114500.txt
```

**包含信息**：
- ✅ 训练时间戳
- ✅ 所有超参数配置
- ✅ 输出文件列表
- ✅ 训练状态（成功/失败）

---

### 4️⃣ **自动清理机制**
```bash
# 自动保留最近 10 个备份，删除旧文件
```

**清理策略**：
- ✅ 保留最近 10 个备份
- ✅ 每次运行后自动清理
- ✅ 节省存储空间

---

## 📊 完整工作流程

### 使用 run_training.bat 的完整流程

```bash
# 1. 运行训练
run_training.bat --epochs 50 --data-limit 100 --batch-size 4

# 2. 查看输出
============================================================
MultiResUNet Training Log
============================================================
Timestamp: 20260303_114500
Training Configuration:
  Epochs: 50
  Batch Size: 4
  Learning Rate: 0.0001
  ...

# 3. 训练完成后
========================================
Training Run Summary
========================================
Log File: runs\logs\training_20260303_114500.log
Models Directory: runs\models\
Training History: runs\histories\history_20260303_114500.npy
Backup File: runs\backup_20260303_114500.tar.gz
Manifest: runs\manifest_20260303_114500.txt
```

### 生成的文件结构

```
runs/
├── logs/
│   └── training_20260303_114500.log      # 完整训练日志
├── models/
│   ├── best_model_checkpoint.pth         # 最佳模型
│   └── model_architecture.pth            # 模型结构
├── histories/
│   └── history_20260303_114500.npy       # 训练指标历史
├── manifest_20260303_114500.txt          # 配置清单
└── backup_20260303_114500.tar.gz         # 完整备份
```

## 🎯 使用示例对比

### 之前（使用 train.py）
```bash
# 运行训练
python train.py --epochs 50 --batch-size 4

# 问题：
# ❌ 日志只在终端显示，关闭后丢失
# ❌ 模型文件散落在项目目录
# ❌ 没有训练历史记录
# ❌ 难以对比不同实验
```

### 现在（使用 run_training.bat）
```bash
# 运行训练
run_training.bat --epochs 50 --batch-size 4

# 优势：
# ✅ 完整日志自动保存
# ✅ 模型集中管理
# ✅ 训练历史可追溯
# ✅ 实验配置清晰记录
# ✅ 自动打包备份
```

## 💡 实际应用场景

### 场景 1: 快速调试
```bash
# 小样本测试，查看日志
run_training.bat --epochs 5 --data-limit 10 --debug

# 查看日志
type runs\logs\training_*.log
```

### 场景 2: 超参数实验
```bash
# 实验 1: 学习率 1e-3
run_training.bat --epochs 50 --learning-rate 1e-3

# 实验 2: 学习率 1e-4
run_training.bat --epochs 50 --learning-rate 1e-4

# 对比结果
findstr "Final Dice" runs\logs\training_*.log
```

### 场景 3: 完整训练
```bash
# 正式训练
run_training.bat --epochs 100 --data-limit 500 --save-model

# 训练完成后
# - 查看日志：type runs\logs\training_TIMESTAMP.log
# - 分析历史：使用 history_TIMESTAMP.npy
# - 提取备份：tar -xzf backup_TIMESTAMP.tar.gz
```

### 场景 4: 批量实验
```bash
# 创建批量实验脚本
cat > run_all.bat << EOF
@echo off
call run_training.bat --epochs 50 --data-limit 100 --learning-rate 1e-3
call run_training.bat --epochs 50 --data-limit 100 --learning-rate 1e-4
call run_training.bat --epochs 50 --data-limit 200 --learning-rate 1e-4
EOF

# 运行所有实验
run_all.bat

# 对比所有结果
for /f "tokens=*" %%i in ('findstr "Final Dice" runs\logs\training_*.log') do echo %%i
```

## 📈 训练历史分析

### 加载和可视化训练历史

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载训练历史
history = np.load('runs/histories/history_20260303_114500.npy', allow_pickle=True).item()

# 绘制训练曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss 曲线
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Training and Validation Loss')

# Dice 系数
axes[1].plot(history['val_dice'], label='Dice', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice Score')
axes[1].set_title('Validation Dice Coefficient')
axes[1].legend()

# Jaccard 指数
axes[2].plot(history['val_jaccard'], label='Jaccard', color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Jaccard Index')
axes[2].set_title('Validation Jaccard Index')
axes[2].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()

# 打印最终指标
print(f"Final Validation Dice: {history['val_dice'][-1]:.4f}")
print(f"Final Validation Jaccard: {history['val_jaccard'][-1]:.4f}")
print(f"Best Validation Dice: {max(history['val_dice']):.4f}")
```

## 🔧 技术实现细节

### 日志记录实现
```batch
# Windows (使用 tee 的替代方案)
python train.py ... 2>&1 | tee -a "%LOG_FILE%"

# Linux (原生支持 tee)
python train.py ... 2>&1 | tee -a "$LOG_FILE"
```

### 备份创建实现
```batch
# Windows (需要 tar 命令，Windows 10+ 支持)
tar -czf "%BACKUP_FILE%" -C "%RUNS_DIR%" ...

# Linux (原生支持)
tar -czf "${BACKUP_FILE}" -C "${RUNS_DIR}" ...
```

### 时间戳生成
```batch
# Windows
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "dt=%%I"
set "TIMESTAMP=%dt:~0,8%_%dt:~8,6%"

# Linux
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
```

## ⚙️ 配置选项

### 通过命令行参数
```bash
run_training.bat --epochs 100 --batch-size 4 --learning-rate 1e-3
```

### 通过环境变量
```batch
# Windows
set EPOCHS=100
set BATCH_SIZE=4
run_training.bat

# Linux
export EPOCHS=100
export BATCH_SIZE=4
bash run_training.sh
```

### 修改默认值
编辑脚本中的默认配置：
```batch
set "EPOCHS=50"           # 改为需要的值
set "BATCH_SIZE=2"
set "LEARNING_RATE=1e-4"
```

## 📋 Manifest 文件示例

```
MultiResUNet Training Run Manifest
===================================
Timestamp: 20260303_114500
Log File: e:\project\ORO\MultiResUNet\runs\logs\training_20260303_114500.log

Configuration:
  Epochs: 50
  Batch Size: 4
  Learning Rate: 1e-4
  Data Limit: 100
  Validation Split: 0.2
  Input Channels: 3
  Output Channels: 4
  Gradient Clip: 1.0
  Device: cuda

Files:
  Log: logs\training_20260303_114500.log
  Models: models\
  History: histories\history_20260303_114500.npy
  Backup: backup_20260303_114500.tar.gz

Training Status: SUCCESS
```

## 🎯 最佳实践推荐

### 1. 开发阶段
```bash
python train.py --epochs 5 --data-limit 10 --debug
```
**原因**：快速迭代，无需等待备份

### 2. 测试阶段
```bash
run_training.bat --epochs 10 --data-limit 20
```
**原因**：开始记录日志，便于调试

### 3. 实验阶段
```bash
run_training.bat --epochs 50 --data-limit 200 --save-model
```
**原因**：完整记录所有实验配置

### 4. 生产阶段
```bash
run_training.bat --epochs 100 --data-limit 500 --save-model
```
**原因**：标准化流程，结果可追溯

## 📊 与 Pytorch-UNet 对比

| 功能 | Pytorch-UNet | MultiResUNet (新增) |
|------|--------------|---------------------|
| 训练脚本 | ✅ run_training.sh | ✅ run_training.bat/sh |
| 日志记录 | ✅ | ✅ |
| 结果打包 | ✅ | ✅ |
| 时间戳 | ✅ | ✅ |
| 自动清理 | ✅ | ✅ |
| 配置清单 | ✅ | ✅ |
| 命令行参数 | ✅ | ✅ |
| 训练历史 | ❌ | ✅ (额外功能) |
| 模型保存 | ✅ | ✅ |
| Manifest 文件 | ❌ | ✅ (额外功能) |

## 🎉 总结

通过参考 `Pytorch-UNet/run_training.sh` 的设计模式，为 MultiResUNet 添加了：

1. ✅ **完整的训练运行脚本**（Windows 和 Linux 版本）
2. ✅ **自动日志记录系统**
3. ✅ **结果自动打包备份**
4. ✅ **配置清单管理**
5. ✅ **自动清理机制**
6. ✅ **详细的使用文档**

现在您可以：
- 🚀 一键启动训练并自动保存所有结果
- 📊 轻松对比不同实验的效果
- 📦 自动备份训练结果便于分享和复现
- 📝 完整的实验记录便于论文写作

**推荐起始命令**：
```bash
# Windows 用户
run_training.bat --epochs 10 --data-limit 20 --debug

# Linux 用户
bash run_training.sh --epochs 10 --data-limit 20 --debug
```
