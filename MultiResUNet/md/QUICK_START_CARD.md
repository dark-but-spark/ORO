# MultiResUNet 训练快速参考卡片 🚀

## ⚡ 立即开始（复制粘贴命令）

### 🔍 调试模式（首次运行推荐）
```bash
# Windows
run_training.bat --epochs 5 --data-limit 20 --batch-size 2 --verbose --debug

# Linux
bash run_training.sh --epochs 5 --data-limit 20 --batch-size 2 --verbose --debug
```

---

### 📊 标准训练（推荐配置）⭐
```bash
# Windows
run_training.bat --epochs 100 --data-limit 500 --batch-size 8 \
  --num-workers 8 --prefetch-factor 3 --gradient-clip 1.0 \
  --save-model --tensorboard

# Linux
bash run_training.sh --epochs 100 --data-limit 500 --batch-size 8 \
  --num-workers 8 --prefetch-factor 3 --gradient-clip 1.0 \
  --save-model --tensorboard
```

---

### 🛡️ 内存安全模式（已发生 OOM 时使用）
```bash
# 超小 batch size
run_training.bat --epochs 50 --data-limit 200 --batch-size 2 \
  --num-workers 6 --gradient-clip 1.0 --debug

# 或限制数据量
run_training.bat --epochs 30 --data-limit 100 --batch-size 4 --verbose
```

---

### 🌙 过夜训练（完整数据集）
```bash
# 使用 screen 防止 SSH 断开
screen -S unet_training

# Linux
bash run_training.sh --epochs 150 --data-limit 2000 \
  --batch-size 4 --num-workers 12 --prefetch-factor 4 \
  --gradient-clip 1.0 --save-model --tensorboard

# 按 Ctrl+A 然后按 D 分离会话
# 重新连接：screen -r unet_training
```

---

## 📈 监控命令

### 实时内存监控
```bash
# Linux
watch -n 1 free -h
nvidia-smi dmon -s puvmet

# Windows
# 任务管理器 → 性能 → 内存/GPU
```

### 查看训练日志
```bash
# 实时查看
tail -f runs/logs/training_*.log

# 搜索警告
grep -i "warning\|error" runs/logs/*.log
```

### TensorBoard 可视化
```bash
tensorboard --logdir runs/tensorboard
# 访问：http://localhost:6006
```

---

## 🎯 参数速查表

### 数据规模 vs 推荐配置

| 样本数 | Batch Size | Workers | Prefetch | 模式 |
|-------|-----------|---------|----------|------|
| <100 | 8 | 4 | 2 | 全量 |
| 100-500 | 8 | 6 | 3 | 混合 |
| >500 | 4-8 | 8-12 | 3-4 | **流式** ⭐ |

### 关键参数说明

```bash
--epochs 100              # 训练轮数
--batch-size 8            # 批次大小（根据显存调整）
--data-limit 500          # 限制样本数量
--num-workers 8           # 数据加载 worker 数量
--prefetch-factor 3       # 预取批次数量
--gradient-clip 1.0       # 梯度裁剪（防止爆炸）
--learning-rate 1e-4      # 初始学习率
--save-model              # 保存最佳模型
--tensorboard             # 启用 TensorBoard
--verbose                 # 详细日志
--debug                   # 调试模式
--check-data              # 数据验证
```

---

## ⚠️ 故障排除速查

### OOM（内存不足）
```bash
# 解决方案：降低配置
run_training.bat --batch-size 2 --data-limit 100
```

### GPU 利用率低
```bash
# 解决方案：增加 workers
run_training.bat --num-workers 12 --prefetch-factor 6
```

### Dice 不收敛
```bash
# 解决方案：调整学习率
run_training.bat --learning-rate 5e-5 --gradient-clip 0.5
```

### 数据加载慢
```bash
# 解决方案：优化 DataLoader
run_training.bat --num-workers 8 --prefetch-factor 4
```

---

## 📁 重要文件位置

```
MultiResUNet/
├── train.py                      # 主训练脚本
├── dataloading.py                # 数据加载模块
├── pytorch/MultiResUNet.py       # 模型定义
├── run_training.bat              # Windows 启动脚本
├── run_training.sh               # Linux 启动脚本
├── OOM_FIX_GUIDE.md              # 完整优化指南
├── OOM_FIX_SUMMARY.md            # 修复总结
└── runs/                         # 训练输出目录
    ├── logs/                     # 训练日志
    ├── models/                   # 保存的模型
    ├── histories/                # 训练历史
    └── tensorboard/              # TensorBoard 日志
```

---

## ✅ 检查清单（每次训练前）

- [ ] 确认数据集大小（决定使用哪种加载模式）
- [ ] 检查可用内存（`free -h` 或任务管理器）
- [ ] 设置合适的 batch size（根据 GPU 显存）
- [ ] 配置 num_workers（CPU 核数的 1/4）
- [ ] 启用梯度裁剪（`--gradient-clip 1.0`）
- [ ] 准备 screen/tmux（长时间训练）
- [ ] 确认输出目录存在（`runs/`）

---

## 🆘 紧急救援命令

```bash
# 强制停止所有 Python 进程
pkill -9 python

# 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 查看占用内存的进程
ps aux | sort -nrk 4 | head -10

# 释放系统缓存（Linux）
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

## 📞 获取帮助

遇到问题？请提供：
1. 完整的训练命令
2. 错误日志（`runs/logs/training_*.log`）
3. 系统配置（内存、GPU、数据规模）

---

**版本**: v2.0 (Memory-Optimized)  
**更新**: 2026-03-09  
**状态**: ✅ 生产就绪
