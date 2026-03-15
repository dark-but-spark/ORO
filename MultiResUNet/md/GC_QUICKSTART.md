# MultiResUNet GC 优化 - 快速开始指南

## 🚀 5 分钟快速上手

### 步骤 1：运行内存诊断（推荐）

```bash
cd e:\project\ORO\MultiResUNet
python diagnose_memory.py
```

**目的**: 检查系统内存、GPU 状态，获取个性化建议

---

### 步骤 2：使用推荐配置训练

#### Windows (PowerShell/CMD):
```cmd
run_training.bat --epochs 150 --data-limit 3500 ^
  --batch-size 4 --num-workers 8 --prefetch-factor 4 ^
  --gradient-clip 1.0 --save-model --tensorboard ^
  --verbose --scale --scale-factor 0.5
```

#### Linux/Mac:
```bash
screen -S unet_gc_v31
bash run_training.sh --epochs 150 --data-limit 3500 \
  --batch-size 4 --num-workers 8 --prefetch-factor 4 \
  --gradient-clip 1.0 --save-model --tensorboard \
  --verbose --scale --scale-factor 0.5
```

---

### 步骤 3：监控训练（另开终端）

#### Windows:
```cmd
powershell -Command "while(1) { Get-Process python | Select-Object Id,CPU,WorkingSet; Start-Sleep -Seconds 10 }"
```

#### Linux/Mac:
```bash
./monitor_memory.sh
```

---

## 📊 预期效果

### 正常输出示例

```
Using device: cuda
GPU Memory: 16.0 GB
✓ Training samples: 3500
✓ Validation samples: 350
✓ Optimized DataLoader config:
  - workers=6 (auto-tuned from 8)
  - prefetch=4
  - persistent_workers=False (memory-safe)
  - pin_memory=True (GPU transfer optimization)

Epoch [1/150], Loss: 0.2345
  📊 GPU Memory Status: Allocated=2150MB, Reserved=2560MB
  Current learning rate: 0.000100
  Validation Dice: 0.7234, Jaccard: 0.5678
  ✓ New best model saved! (Dice: 0.7234)

Epoch [5/150], Loss: 0.1823
  📊 GPU Memory Status: Allocated=2180MB, Reserved=2560MB
  ...
```

### 关键指标

✅ **正常状态**:
- GPU 内存稳定在 2-3 GB
- RSS 内存 < 500 MB
- GPU 利用率 > 75%
- Dice 逐步上升

⚠️ **警告信号**:
- GPU 内存 > 6 GB（会触发预警）
- RSS 内存持续增长
- GPU 利用率 < 50%
- dmesg 出现 OOM kill

---

## 🎯 根据数据规模选择配置

### 小规模 (<500 样本)

```bash
--epochs 50 --data-limit 300 --batch-size 8 \
--num-workers 6 --prefetch-factor 3 --verbose
```

### 中等规模 (500-2000 样本)

```bash
--epochs 100 --data-limit 1000 --batch-size 8 \
--num-workers 8 --prefetch-factor 4 \
--gradient-clip 1.0 --save-model --verbose
```

### 大规模 (>2000 样本) ⭐ 推荐

```bash
--epochs 150 --data-limit 3500 --batch-size 4 \
--num-workers 8 --prefetch-factor 4 \
--gradient-clip 1.0 --save-model --tensorboard \
--verbose --scale --scale-factor 0.5
```

### 超大规模 (>5000 样本)

```bash
--epochs 200 --data-limit 5000 --batch-size 2 \
--num-workers 6 --prefetch-factor 3 \
--gradient-clip 1.0 --weight-decay 1e-5 \
--scale --scale-factor 0.5
```

---

## 🆘 紧急情况处理

### 场景 1：内存占用过高（>8GB）

**立即执行**:
```bash
# Windows
Ctrl+C  # 停止训练

# Linux
pkill -9 python
```

**调整后重启**:
```bash
--batch-size 2 --num-workers 4 --prefetch-factor 2
```

---

### 场景 2：训练突然停止

**检查原因**:
```bash
# 查看是否被 OOM kill
dmesg -T | grep -i "kill" | tail -5
```

**恢复训练**:
```bash
# 从上次保存点继续（如果支持断点续训）
# 或降低配置重新开始
```

---

## 📋 验证清单

### 训练前 ✓
- [ ] 已运行 `diagnose_memory.py`
- [ ] 确认可用内存 >10GB
- [ ] 创建了 screen 会话（Linux）
- [ ] 准备了监控脚本

### 训练中 ✓
- [ ] 内存稳定在 200-500 MB
- [ ] GPU 利用率 >75%
- [ ] 无 dmesg kill 记录
- [ ] Dice 系数上升

### 训练后 ✓
- [ ] 模型已保存至 `runs/models/`
- [ ] 日志完整（`runs/logs/`）
- [ ] TensorBoard 可访问
- [ ] 无 OOM 记录

---

## 🔧 常用命令速查

### 监控命令
```bash
# 系统内存
free -h

# GPU 状态
nvidia-smi

# 进程内存
ps aux | grep train.py

# OOM 检查
dmesg -T | grep -i "kill"
```

### 管理命令
```bash
# 创建 screen 会话
screen -S unet_training

# 分离会话
Ctrl+A, D

# 重新连接
screen -r unet_training

# 列出会话
screen -ls

# 杀死进程
pkill -f train.py
```

---

## 📞 获取帮助

### 查看完整文档
- **详细优化说明**: `GC_OPTIMIZATION_SUMMARY.md`
- **快速参考**: `GC_OPTIMIZATION_QUICKREF.md`
- **技术方案**: `AGGRESSIVE_GC_OPTIMIZATION.md`

### 查看所有参数
```bash
python train.py --help
```

### 常见问题
详见 `GC_OPTIMIZATION_QUICKREF.md` 的故障排除部分

---

## ✅ 成功标准

训练成功的标志：
1. ✅ 完成所有 epochs
2. ✅ 内存始终 < 500 MB
3. ✅ 无 OOM 错误
4. ✅ Dice 系数 > 0.8
5. ✅ 模型已保存

---

**版本**: v3.1 Enhanced GC  
**更新**: 2026-03-10  
**适用**: MultiResUNet 大规模数据集训练

**下一步**: 
1. 运行 `python diagnose_memory.py`
2. 选择合适的配置
3. 开始训练并监控
4. 享受零 OOM 的训练体验！🎉
