# 错误日志捕获功能

## ✅ 自动错误记录（推荐）

使用增强版的 `run_training.sh` 脚本，会自动将错误保存到单独的文件：

```bash
bash run_training.sh --epochs 150 --data-limit 3500 \
  --batch-size 4 --num-workers 8
```

**输出文件**：
- `runs/logs/training_20260311_123456.log` - 标准输出（正常日志）
- `runs/logs/training_20260311_123456.err` - 错误输出（Python 异常、Traceback 等）

---

## 🔧 手动重定向方法

### 方法 1：分别保存输出和错误

```bash
# 标准输出到 .out，错误输出到 .err
bash run_training.sh --epochs 150 > training.out 2> training.err

# 查看错误
cat training.err

# 实时查看错误
tail -f training.err
```

### 方法 2：合并到一个文件

```bash
# 所有输出（含错误）都保存到 .log
bash run_training.sh --epochs 150 > training.log 2>&1

# 或者使用 tee 同时显示和保存
bash run_training.sh --epochs 150 2>&1 | tee training.log
```

### 方法 3：后台运行 + 错误捕获

```bash
# 使用 nohup 后台运行
nohup bash run_training.sh --epochs 150 > training.log 2>&1 &

# 查看进程
ps aux | grep train.py

# 查看错误
tail -f training.log | grep -i "error\|exception"
```

### 方法 4：使用 screen 会话（最佳实践）

```bash
# 创建带日志的 screen 会话
screen -L -Logfile training_$(date +%Y%m%d_%H%M%S).log -S unet_training

# 运行训练
bash run_training.sh --epochs 150 --data-limit 3500

# 分离会话：Ctrl+A 然后 D
# 重新连接：screen -r unet_training
```

---

## 📊 错误日志分析

### 查看最近的错误

```bash
# 最新的错误文件
ls -lt runs/logs/*.err | head -1

# 查看最后 50 行错误
tail -50 runs/logs/training_*.err

# 查看完整的错误堆栈
cat runs/logs/training_*.err
```

### 搜索特定错误类型

```bash
# Python Traceback
grep -A 20 "Traceback" runs/logs/training_*.err

# CUDA 错误
grep -i "cuda\|GPU\|OOM" runs/logs/training_*.err

# 内存错误
grep -i "memory\|OOM\|killed" runs/logs/training_*.err

# NaN 或 Inf
grep -i "nan\|inf" runs/logs/training_*.err
```

### 实时监控错误

```bash
# 实时显示新出现的错误
tail -f runs/logs/training_*.err

# 只显示包含 error 的行
tail -f runs/logs/training_*.err | grep -i "error"
```

---

## 🆘 故障排查流程

### 步骤 1：检查训练状态

```bash
# 查看是否还在运行
ps aux | grep train.py

# 查看退出码
echo $?  # 0=成功，非 0=失败
```

### 步骤 2：查看错误日志

```bash
# 找到最新的错误文件
ERR_FILE=$(ls -t runs/logs/*.err | head -1)

# 查看完整错误
cat "$ERR_FILE"

# 或分段查看
head -100 "$ERR_FILE"    # 前 100 行
tail -100 "$ERR_FILE"    # 后 100 行
```

### 步骤 3：分析常见错误

#### OOM 错误
```
RuntimeError: CUDA out of memory
```
**解决**：降低 batch size 或使用 scale 缩放

#### NaN 损失
```
Loss became NaN or Inf
```
**解决**：降低学习率或启用梯度裁剪

#### 数据加载错误
```
FileNotFoundError: [Errno 2] No such file or directory
```
**解决**：检查数据路径配置

---

## 💡 最佳实践建议

1. **始终使用增强版脚本** - 自动分离错误日志
2. **定期检查 .err 文件** - 即使训练成功也可能有警告
3. **保留历史日志** - 便于对比不同实验
4. **使用 screen/tmux** - 防止 SSH 断开丢失输出
5. **设置日志轮转** - 避免单个文件过大

---

## 📋 快速命令参考

```bash
# 启动训练并记录错误
bash run_training.sh --epochs 150 --data-limit 3500

# 实时查看错误
tail -f runs/logs/*.err

# 查找最新错误文件
find runs/logs -name "*.err" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-

# 提取错误堆栈
grep -A 30 "Traceback" runs/logs/training_*.err

# 清理旧日志（保留最近 10 个）
cd runs/logs && ls -t *.err | tail -n +11 | xargs rm -f
```

---

**更新时间**: 2026-03-11  
**适用版本**: MultiResUNet v3.1+
