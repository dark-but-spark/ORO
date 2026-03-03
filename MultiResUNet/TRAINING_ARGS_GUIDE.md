# MultiResUNet 训练命令行参数使用指南

## 快速开始

### 基本训练（默认配置）
```bash
python train.py
```

这将使用默认配置开始训练：
- 100 个训练样本
- 50 个 epoch
- batch size = 2
- 学习率 = 1e-4

## 命令行参数详解

### 数据加载参数

#### `--data-limit`
控制加载的训练样本数量。

**用途**：快速调试或测试时使用少量样本。

**示例**：
```bash
# 只使用 10 个样本进行快速测试
python train.py --data-limit 10

# 使用全部数据（假设数据集中有更多样本）
python train.py --data-limit 1000
```

**默认值**：100

---

#### `--validation-split`
设置验证集占总数据的比例。

**示例**：
```bash
# 使用 30% 数据作为验证集
python train.py --validation-split 0.3

# 使用 10% 数据作为验证集
python train.py --validation-split 0.1
```

**默认值**：0.2 (20%)

---

### 模型参数

#### `--input-channels`
输入图像的通道数。

**示例**：
```bash
# RGB 图像（默认）
python train.py --input-channels 3

# 灰度图像
python train.py --input-channels 1
```

**默认值**：3

---

#### `--output-channels`
输出分割掩码的通道数。

**示例**：
```bash
# 4 通道分割（默认）
python train.py --output-channels 4

# 单通道二值分割
python train.py --output-channels 1
```

**默认值**：4

---

### 训练超参数

#### `--epochs`
训练的总轮次数。

**示例**：
```bash
# 训练 100 个 epoch
python train.py --epochs 100

# 快速测试训练 5 个 epoch
python train.py --epochs 5
```

**默认值**：50

---

#### `--batch-size`
每个批次的样本数。

**示例**：
```bash
# 小批量（适合显存小的 GPU）
python train.py --batch-size 1

# 大批量（适合显存充足的 GPU）
python train.py --batch-size 8
```

**默认值**：2

---

#### `--learning-rate`
初始学习率。

**示例**：
```bash
# 较高的学习率（加速收敛，但可能不稳定）
python train.py --learning-rate 1e-3

# 较低的学习率（稳定但收敛慢）
python train.py --learning-rate 5e-5
```

**默认值**：1e-4

---

### 优化参数

#### `--gradient-clip`
梯度裁剪的最大范数。用于防止梯度爆炸。

**示例**：
```bash
# 更强的梯度裁剪
python train.py --gradient-clip 0.5

# 禁用梯度裁剪
python train.py --gradient-clip 0
```

**默认值**：1.0

---

#### `--weight-decay`
权重衰减（L2 正则化），用于防止过拟合。

**示例**：
```bash
# 添加正则化
python train.py --weight-decay 1e-5

# 不使用正则化（默认）
python train.py --weight-decay 0
```

**默认值**：0

---

### 日志和保存参数

#### `--verbose`
启用详细日志输出。

**示例**：
```bash
# 显示更多训练细节
python train.py --verbose
```

**默认值**：关闭

---

#### `--save-model`
保存训练过程中的最佳模型检查点。

**示例**：
```bash
# 保存最佳模型
python train.py --save-model

# 保存到指定目录
python train.py --save-model --save-dir my_models
```

**默认值**：关闭

---

#### `--save-dir`
模型保存的目录。

**示例**：
```bash
python train.py --save-model --save-dir experiments/exp1
```

**默认值**：`models`

---

### 调试选项

#### `--debug`
启用调试模式，输出详细的数据和训练信息。

**示例**：
```bash
# 完整调试模式
python train.py --debug --verbose

# 调试 + 保存模型
python train.py --debug --save-model
```

**默认值**：关闭

---

#### `--check-data`
在训练前运行数据验证检查。

**示例**：
```bash
# 只验证数据，不训练
python train.py --check-data --data-limit 5

# 验证数据并训练
python train.py --check-data
```

**默认值**：关闭

---

### 设备选择

#### `--device`
选择训练设备（CPU 或 GPU）。

**示例**：
```bash
# 使用 GPU（默认，如果有可用 GPU）
python train.py --device cuda

# 强制使用 CPU
python train.py --device cpu
```

**默认值**：`cuda`（自动检测）

---

## 常用命令组合

### 1. 快速测试配置
用于快速验证代码是否能正常运行。

```bash
python train.py --data-limit 10 --epochs 5 --batch-size 2
```

### 2. 调试模式
用于详细分析训练过程和数据问题。

```bash
python train.py --debug --verbose --check-data --save-model
```

### 3. 小规模实验
用于在完整训练前测试超参数配置。

```bash
python train.py --data-limit 100 --epochs 20 --batch-size 4 --learning-rate 1e-3
```

### 4. 完整训练
使用较多数据和 epoch 进行正式训练。

```bash
python train.py --data-limit 500 --epochs 100 --batch-size 4 --save-model
```

### 5. 过拟合测试
使用极少数据测试模型是否能过拟合（验证模型容量）。

```bash
python train.py --data-limit 5 --epochs 100 --batch-size 1
```

### 6. 学习率搜索
测试不同学习率的效果。

```bash
# 测试高学习率
python train.py --learning-rate 1e-3 --epochs 30 --data-limit 100

# 测试低学习率
python train.py --learning-rate 5e-5 --epochs 30 --data-limit 100
```

### 7. 梯度稳定性测试
当训练不稳定时，尝试更强的梯度裁剪。

```bash
python train.py --gradient-clip 0.5 --learning-rate 5e-5 --debug
```

---

## 预期输出示例

### 正常训练输出
```
============================================================
MultiResUNet Training Configuration
============================================================
Data Limit: 100 samples
Validation Split: 20.0%
Input Channels: 3
Output Channels: 4
Epochs: 50
Batch Size: 2
Learning Rate: 0.0001
Gradient Clipping: 1.0
Weight Decay: 0.0
Device: cuda
Debug Mode: False
Data Validation: False
============================================================
Using device: cuda

Loading data (limit=100)...
Number of image files: 100
Number of label files: 100
...

Splitting data (validation=20.0%)...
Train set: (80, 640, 640, 3)
Validation set: (20, 640, 640, 3)

Initializing model...
Model architecture: MultiResUNet
  Input: 3 channels
  Output: 4 channels
Training samples: 80, Validation samples: 20

Starting training...
------------------------------------------------------------
Y_train range: [0.0000, 1.0000]
Y_train unique values: tensor([0., 1.], device='cuda:0')
Y_train positive pixel ratio: 0.0193
Epoch [1/50], Loss: 0.6523
Average Dice Coefficient: 0.1523
Average Jaccard Index: 0.0845
  Current learning rate: 0.000100
  Validation Dice: 0.1523, Jaccard: 0.0845
...
```

### 调试模式输出
```
============================================================
MultiResUNet Training Configuration
============================================================
...
Debug Mode: True
Data Validation: True
============================================================

Data Validation:
  Original Y shape: (100, 640, 640, 4)
  Y sample unique values: [0. 1.]
  Y value range: [0.0000, 1.0000]
  Y positive pixel ratio: 0.0193

Data Statistics:
  X_train range: [0.0000, 1.0000]
  Y_train range: [0.0000, 1.0000]
  Y_train positive ratio: 0.0193
...
```

---

## 故障排查

### 问题 1: 训练不收敛
**尝试**：
```bash
python train.py --learning-rate 1e-3 --gradient-clip 0.5 --debug
```

### 问题 2: 显存不足
**尝试**：
```bash
python train.py --batch-size 1 --data-limit 50
```

### 问题 3: 训练太慢
**尝试**：
```bash
python train.py --batch-size 8 --data-limit 50 --epochs 20
```

### 问题 4: 查看数据是否有问题
**尝试**：
```bash
python train.py --check-data --debug --data-limit 10
```

---

## 帮助信息

查看所有可用参数：
```bash
python train.py --help
```

输出：
```
usage: train.py [-h] [--data-limit DATA_LIMIT] [--validation-split VALIDATION_SPLIT]
                [--input-channels INPUT_CHANNELS] [--output-channels OUTPUT_CHANNELS]
                [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                [--gradient-clip GRADIENT_CLIP] [--weight-decay WEIGHT_DECAY]
                [--verbose] [--save-model] [--save-dir SAVE_DIR]
                [--debug] [--check-data] [--device {cuda,cpu}]

Train MultiResUNet for image segmentation

optional arguments:
  -h, --help            show this help message and exit
  --data-limit DATA_LIMIT
                        Number of samples to load for training (default: 100)
  --validation-split VALIDATION_SPLIT
                        Proportion of data used for validation (default: 0.2)
  --input-channels INPUT_CHANNELS
                        Number of input image channels (default: 3)
  --output-channels OUTPUT_CHANNELS
                        Number of output segmentation channels (default: 4)
  --epochs EPOCHS       Number of training epochs (default: 50)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 2)
  --learning-rate LEARNING_RATE
                        Initial learning rate (default: 1e-4)
  --gradient-clip GRADIENT_CLIP
                        Maximum gradient norm for clipping (default: 1.0)
  --weight-decay WEIGHT_DECAY
                        Weight decay (L2 regularization) for optimizer (default: 0)
  --verbose             Enable verbose logging during training
  --save-model          Save model checkpoints during training
  --save-dir SAVE_DIR   Directory to save model checkpoints (default: models)
  --debug               Enable debug mode with additional logging
  --check-data          Run data validation checks before training
  --device {cuda,cpu}   Device to use for training (default: cuda)
```

---

## 最佳实践建议

1. **初次运行**：先用小数据集测试
   ```bash
   python train.py --data-limit 10 --epochs 5
   ```

2. **调试数据**：使用 `--check-data` 和 `--debug` 验证数据

3. **超参数调优**：从默认值开始，逐步调整学习率和 batch size

4. **保存最佳模型**：正式训练时始终使用 `--save-model`

5. **监控训练**：使用 `--verbose` 查看详细训练过程

6. **GPU 加速**：确保使用 `--device cuda`（如果有 GPU）
