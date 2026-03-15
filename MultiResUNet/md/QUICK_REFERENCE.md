# MultiResUNet 训练快速参考卡片

## 🚀 快速开始命令

### Windows 用户
```bash
# 1. 快速测试（推荐首次使用）
run_training.bat --epochs 5 --data-limit 10

# 2. 正式训练
run_training.bat --epochs 50 --data-limit 100

# 3. 查看帮助
run_training.bat --help
```

### Linux/Mac 用户
```bash
# 1. 快速测试
bash run_training.sh --epochs 5 --data-limit 10

# 2. 正式训练
bash run_training.sh --epochs 50 --data-limit 100

# 3. 查看帮助
bash run_training.sh --help
```

## 📊 常用参数速查

| 参数 | 简写 | 默认值 | 示例 |
|------|------|--------|------|
| `--epochs` | -e | 50 | `--epochs 100` |
| `--batch-size` | -b | 2 | `--batch-size 4` |
| `--learning-rate` | -lr | 1e-4 | `--learning-rate 1e-3` |
| `--data-limit` | -dl | 100 | `--data-limit 200` |
| `--device` | -d | cuda | `--device cpu` |

## 🎯 典型场景命令

### 场景 1: 快速验证
```bash
run_training.bat -e 5 -dl 10
```

### 场景 2: 调试模式
```bash
run_training.bat -e 10 -dl 20 --debug --verbose
```

### 场景 3: 完整训练
```bash
run_training.bat -e 100 -dl 500 -b 8 --save-model
```

### 场景 4: 学习率实验
```bash
run_training.bat -e 50 -dl 200 -lr 1e-3
```

### 场景 5: 显存不足
```bash
run_training.bat -b 1 -dl 50
```

## 📁 输出文件位置

```
runs/
├── logs/           # 训练日志
├── models/         # 模型文件
├── histories/      # 训练历史
├── manifest_*.txt  # 配置清单
└── backup_*.tar.gz # 备份文件
```

## 🔍 查看结果

### 查看日志
```bash
# Windows
type runs\logs\training_*.log

# Linux
cat runs/logs/training_*.log
```

### 查看备份
```bash
# Windows
dir runs\backup_*.tar.gz

# Linux
ls -lh runs/backup_*.tar.gz
```

### 解压备份
```bash
tar -xzf backup_20260303_114500.tar.gz -C your_destination
```

## 📈 分析训练历史

```python
import numpy as np

# 加载历史
history = np.load('runs/histories/history_TIMESTAMP.npy', 
                  allow_pickle=True).item()

# 查看最终结果
print(f"Final Dice: {history['val_dice'][-1]:.4f}")
print(f"Final Jaccard: {history['val_jaccard'][-1]:.4f}")
```

## ⚡ 快捷键/技巧

### 1. 使用环境变量
```batch
# Windows (PowerShell)
$env:EPOCHS=100
run_training.bat

# Linux
export EPOCHS=100
bash run_training.sh
```

### 2. 批量实验
```batch
@echo off
call run_training.bat -e 50 -dl 100 -lr 1e-3
call run_training.bat -e 50 -dl 100 -lr 1e-4
call run_training.bat -e 50 -dl 200 -lr 1e-4
```

### 3. 对比实验
```bash
# Windows
findstr "Final Dice" runs\logs\training_*.log

# Linux
grep "Final Dice" runs/logs/training_*.log
```

## 🎯 推荐工作流

```
1. 快速测试
   run_training.bat -e 5 -dl 10

2. 调试数据
   run_training.bat -e 10 -dl 20 --debug

3. 小规模实验
   run_training.bat -e 20 -dl 50

4. 正式训练
   run_training.bat -e 100 -dl 500 --save-model

5. 分析结果
   使用 Python 脚本分析 history_*.npy
```

## ⚠️ 常见问题

### Q: 训练失败？
```bash
# 查看详细日志
type runs\logs\training_TIMESTAMP.log
```

### Q: 显存不足？
```bash
# 减小 batch size
run_training.bat -b 1 -dl 50
```

### Q: 训练太慢？
```bash
# 减少数据量
run_training.bat -dl 50 -e 20
```

### Q: 找不到备份文件？
```bash
# 检查 runs 目录
dir runs\
```

## 📚 详细文档

- `RUN_TRAINING_GUIDE.md` - 完整使用指南
- `SCRIPT_COMPARISON.md` - 脚本对比
- `TRAINING_SCRIPT_SUMMARY.md` - 功能总结

## 🎉 开始训练

```bash
# 立即开始
run_training.bat --epochs 10 --data-limit 20 --debug
```

---

**提示**：将此文件保存为 `QUICK_REFERENCE.md` 便于快速查阅！
