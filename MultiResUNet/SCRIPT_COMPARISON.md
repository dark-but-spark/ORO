# 训练脚本对比与选择指南

## 📋 脚本列表

MultiResUNet 项目现在提供三种训练方式：

1. **`train.py`** - Python 训练脚本（基础）
2. **`run_training.bat`** - Windows 批处理包装脚本
3. **`run_training.sh`** - Linux/Mac Shell 包装脚本

## 🎯 使用场景对比

| 使用场景 | 推荐脚本 | 原因 |
|---------|----------|------|
| 快速测试 | `train.py` | 直接运行，无需额外包装 |
| 正式训练 | `run_training.bat/sh` | 自动日志和备份 |
| 实验管理 | `run_training.bat/sh` | 完整的实验记录和打包 |
| 调试模式 | `train.py` | 更灵活的参数调整 |
| 生产环境 | `run_training.bat/sh` | 标准化流程和结果保存 |

##  功能对比

| 功能 | train.py | run_training.bat/sh |
|------|----------|---------------------|
| 训练模型 | ✅ | ✅ |
| 命令行参数 | ✅ | ✅ |
| 日志文件 | ❌ | ✅ 自动保存 |
| 结果打包 | ❌ | ✅ 自动备份 |
| 时间戳管理 | ❌ | ✅ 自动创建 |
| 配置清单 | ❌ | ✅ manifest 文件 |
| 自动清理 | ❌ | ✅ 保留最近 10 个 |
| 训练历史 | ✅ | ✅ 自动保存 |
| 模型保存 | ✅ | ✅ 自动复制 |
| 跨平台 | ✅ | ⚠️ 需要不同版本 |

## 🚀 快速选择指南

### 选择 `train.py` 如果：
- ✅ 只需要快速测试代码
- ✅ 在 IDE 中调试
- ✅ 不需要自动备份
- ✅ 习惯手动管理日志

**示例**：
```bash
python train.py --epochs 5 --data-limit 10 --debug
```

### 选择 `run_training.bat/sh` 如果：
- ✅ 进行正式训练
- ✅ 需要完整的实验记录
- ✅ 希望自动保存所有结果
- ✅ 需要对比多个实验
- ✅ 生产环境部署

**示例**：
```bash
# Windows
run_training.bat --epochs 100 --data-limit 500

# Linux
bash run_training.sh --epochs 100 --data-limit 500
```

## 📁 输出文件对比

### 使用 `train.py`
```
项目目录/
├── models/                    # 模型文件（如果指定 --save-model）
├── training_history.npy       # 训练历史
└── 终端输出（不保存）
```

### 使用 `run_training.bat/sh`
```
runs/
├── logs/
│   └── training_TIMESTAMP.log    # 完整训练日志
├── models/
│   └── best_model_checkpoint.pth # 最佳模型
├── histories/
│   └── history_TIMESTAMP.npy     # 训练历史
├── manifest_TIMESTAMP.txt        # 配置清单
└── backup_TIMESTAMP.tar.gz       # 完整备份
```

## 💡 推荐工作流程

### 阶段 1: 开发和测试
```bash
# 使用 train.py 快速迭代
python train.py --epochs 5 --data-limit 10 --debug
```

### 阶段 2: 小规模实验
```bash
# 使用 run_training.bat/sh 开始记录
run_training.bat --epochs 20 --data-limit 50
```

### 阶段 3: 正式训练
```bash
# 使用 run_training.bat/sh 完整记录
run_training.bat --epochs 100 --data-limit 500 --save-model
```

### 阶段 4: 结果分析
```python
# 使用保存的文件分析
import numpy as np
history = np.load('runs/histories/history_TIMESTAMP.npy', allow_pickle=True)
```

## 🎯 典型使用模式

### 模式 1: 单次训练
```bash
# 简单直接
run_training.bat --epochs 50 --batch-size 4
```

### 模式 2: 参数扫描
```bash
# 测试不同学习率
run_training.bat --epochs 50 --learning-rate 1e-3
run_training.bat --epochs 50 --learning-rate 1e-4
run_training.bat --epochs 50 --learning-rate 5e-5
```

### 模式 3: 数据量实验
```bash
# 测试不同数据量
run_training.bat --data-limit 50 --epochs 30
run_training.bat --data-limit 100 --epochs 30
run_training.bat --data-limit 200 --epochs 30
```

### 模式 4: 批量实验
```bash
# 创建实验脚本
cat > run_experiments.bat << EOF
@echo off
call run_training.bat --epochs 50 --data-limit 100 --learning-rate 1e-3
call run_training.bat --epochs 50 --data-limit 100 --learning-rate 1e-4
call run_training.bat --epochs 50 --data-limit 200 --learning-rate 1e-4
EOF

# 运行所有实验
run_experiments.bat
```

## 📖 实际案例

### 案例 1: 快速验证
**目标**：验证代码是否能正常运行

```bash
python train.py --data-limit 5 --epochs 3
```

**预期**：3-5 分钟内完成，确认无错误

---

### 案例 2: 调试数据问题
**目标**：检查数据加载是否正确

```bash
run_training.bat --data-limit 10 --epochs 5 --debug --check-data --verbose
```

**输出**：
- `runs/logs/training_TIMESTAMP.log` - 详细日志
- `runs/manifest_TIMESTAMP.txt` - 配置信息

---

### 案例 3: 超参数调优
**目标**：找到最佳学习率

```bash
# 实验 1: 高学习率
run_training.bat --epochs 30 --data-limit 100 --learning-rate 1e-3

# 实验 2: 中等学习率
run_training.bat --epochs 30 --data-limit 100 --learning-rate 1e-4

# 实验 3: 低学习率
run_training.bat --epochs 30 --data-limit 100 --learning-rate 5e-5

# 对比结果
cat runs/logs/training_*.log | findstr "Final Dice"
```

---

### 案例 4: 完整训练流程
**目标**：训练最终模型用于论文/项目

```bash
# 第一阶段：小规模测试
run_training.bat --epochs 10 --data-limit 50

# 第二阶段：中等规模训练
run_training.bat --epochs 50 --data-limit 200

# 第三阶段：完整训练
run_training.bat --epochs 100 --data-limit 500 --save-model

# 查看结果
dir runs\backups
```

## 🔍 结果对比方法

### 对比日志
```bash
# Windows
type runs\logs\training_*.log | findstr "Dice"

# Linux
grep "Dice" runs/logs/training_*.log
```

### 对比配置
```bash
# 查看 manifest 文件
cat runs/manifest_*.txt
```

### 对比训练曲线
```python
import numpy as np
import matplotlib.pyplot as plt

# 加载多个实验的历史
exp1 = np.load('runs/histories/history_001.npy', allow_pickle=True).item()
exp2 = np.load('runs/histories/history_002.npy', allow_pickle=True).item()

# 对比
plt.plot(exp1['val_dice'], label='Exp 1')
plt.plot(exp2['val_dice'], label='Exp 2')
plt.legend()
plt.show()
```

## ⚙️ 自定义配置

### 方法 1: 修改脚本默认值
编辑 `run_training.bat` 或 `run_training.sh`，修改默认参数：
```bash
# 在脚本中找到这些行
set "EPOCHS=50"           # 改为 set "EPOCHS=100"
set "BATCH_SIZE=2"        # 改为 set "BATCH_SIZE=4"
set "DATA_LIMIT=100"      # 改为 set "DATA_LIMIT=200"
```

### 方法 2: 使用环境变量
```bash
# Windows (PowerShell)
$env:EPOCHS=100
$env:BATCH_SIZE=4
.\run_training.bat

# Linux
export EPOCHS=100
export BATCH_SIZE=4
bash run_training.sh
```

### 方法 3: 创建自定义脚本
```bash
# 创建自己的实验脚本
cat > my_experiment.bat << EOF
@echo off
call run_training.bat --epochs 100 --data-limit 500 --batch-size 8 --learning-rate 5e-5
EOF

# 运行
my_experiment.bat
```

## 📊 性能对比

| 训练规模 | train.py | run_training.bat/sh |
|---------|----------|---------------------|
| 小 (10 样本，5 epochs) | ~1 分钟 | ~1 分钟 |
| 中 (100 样本，50 epochs) | ~10 分钟 | ~10 分钟 |
| 大 (500 样本，100 epochs) | ~2 小时 | ~2 小时 |

**注意**：包装脚本只增加几秒钟的初始化和打包时间

## 🎉 总结推荐

### 日常开发
使用 `train.py`，灵活快速

### 正式实验
使用 `run_training.bat/sh`，完整记录

### 生产部署
使用 `run_training.bat/sh`，标准化流程

### 论文实验
使用 `run_training.bat/sh`，便于复现

---

**最佳实践**：
1. 开发阶段：`train.py`
2. 测试阶段：`run_training.bat --epochs 5 --data-limit 10`
3. 实验阶段：`run_training.bat --epochs 50 --data-limit 200`
4. 生产阶段：`run_training.bat --epochs 100 --data-limit 500 --save-model`
