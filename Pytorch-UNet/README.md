# U-Net: PyTorch 语义分割实现（本地化增强版）

<a href="#"><img src="https://img.shields.io/github/actions/workflow/status/milesial/PyTorch-UNet/main.yml?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)

## 📋 目录

- [简介](#简介)
- [快速开始](#快速开始)
- [主要特性](#主要特性)
- [安装指南](#安装指南)
- [使用教程](#使用教程)
  - [数据准备](#数据准备)
  - [训练模型](#训练模型)
  - [预测与评估](#预测与评估)
- [高级功能](#高级功能)
- [故障排查](#故障排查)
- [性能优化](#性能优化)
- [引用](#引用)

---

## 简介

这是 [U-Net](https://arxiv.org/abs/1505.04597) 的 PyTorch 实现，基于 Kaggle [Carvana 图像掩码挑战赛](https://www.kaggle.com/c/carvana-image-masking-challenge) 的高清图像进行定制。

**核心优势**:
- ✅ 在 100k+ 测试图像上达到 **0.988423** Dice 系数
- ✅ 支持多类别分割、人像分割、医学图像分割
- ✅ 完整的训练、预测、评估工具链
- ✅ TensorBoard 实时监控
- ✅ 自动化备份和日志管理
- ✅ 完善的诊断和调试工具

---

## 快速开始

### 无需 Docker

```bash
# 1. 安装 CUDA (可选，用于 GPU 加速)
# 访问 https://developer.nvidia.com/cuda-downloads

# 2. 安装 PyTorch
pip install torch torchvision

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载数据并训练
bash scripts/download_data.sh
python train.py --amp
```

### 使用 Docker

```bash
# 1. 安装 Docker
curl https://get.docker.com | sh && sudo systemctl --now enable docker

# 2. 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. 运行容器
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

---

## 主要特性

### 🎯 核心功能
- **多通道支持**: 自动检测和处理多通道掩码（如 4 通道二值掩码）
- **自动混合精度**: `--amp` 参数启用，节省内存提升速度
- **灵活的数据加载**: 支持 PNG、NPZ 等多种格式
- **实时可视化**: TensorBoard 集成，监控损失和指标

### 🔧 增强功能
- **自动化备份**: 训练结束自动生成模型、日志、配置备份
- **智能日志管理**: 按时间戳分离保存，保留最近 10 个备份
- **诊断工具集**: 
  - `debug_training.py` - 数据加载验证
  - `mask_analysis.py` - 掩码格式分析
  - `comprehensive_diagnostics.sh` - 综合诊断

### 📊 训练配置
支持丰富的超参数调整：
```bash
python train.py --epochs 100 --batch-size 4 --learning-rate 1e-4 \
  --data-limit 500 --validation-split 0.2 \
  --input-channels 3 --output-channels 4 \
  --gradient-clip 1.0 --weight-decay 1e-5 \
  --amp --tensorboard --save-model --verbose
```

---

## 安装指南

### 系统要求
- Python 3.6+
- PyTorch 1.13+ (推荐 2.0+)
- CUDA 11.0+ (GPU 加速，可选)
- Linux/MacOS/Windows

### 安装步骤

**方法 1: pip 安装**
```bash
# 克隆或进入项目目录
cd Pytorch-UNet

# 安装核心依赖
pip install torch torchvision numpy tqdm tensorboard pillow

# 可选：Weights & Biases
pip install wandb
```

**方法 2: Conda 环境（推荐）**
```bash
# 创建环境
conda create -n unet python=3.10 -y
conda activate unet

# 安装 PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy tqdm tensorboard pillow matplotlib
```

---

## 使用教程

### 数据准备

#### 数据结构
```
data/
├── imgs/           # 输入图像（RGB 或灰度）
│   ├── img1.png
│   └── img2.jpg
└── masks/          # 对应掩码
    ├── img1.png
    └── img2.png
```

**要求**:
- 图像和掩码文件名对应
- 掩码可以是：
  - 单通道灰度图（0-255）
  - 多通道 NPZ 文件（[H, W, C]）
  - 二值图（0 和 255）

#### 下载示例数据
```bash
bash scripts/download_data.sh
```

#### 检查数据
```bash
# 运行数据诊断
python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/

# 或使用综合诊断脚本
bash quick_diagnostics.sh
```

### 训练模型

#### 基础训练
```bash
# 默认配置（scale=0.5, AMP 启用）
python train.py --amp

# 指定 epochs 和 batch size
python train.py --epochs 50 --batch-size 8 --amp
```

#### 完整参数说明
```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL]
                [--data-limit N] [--num-workers N] [--prefetch-factor N]
                [--input-channels N] [--output-channels N]
                [--gradient-clip V] [--weight-decay V]
                [--amp] [--tensorboard] [--save-model]
                [--verbose] [--debug] [--device DEVICE]

训练 UNet 模型

optional arguments:
  -h, --help            显示帮助信息
  --epochs E, -e E      训练轮数 (默认：100)
  --batch-size B, -b B  批次大小 (默认：4)
  --learning-rate LR, -l LR
                        学习率 (默认：1e-4)
  --load LOAD, -f LOAD  从.pth 文件加载模型
  --scale SCALE, -s SCALE
                        图像缩放因子 (默认：0.5)
  --validation VAL, -v VAL
                        验证集比例 (0-100, 默认：20)
  --data-limit N        限制训练数据量 (用于调试)
  --num-workers N       DataLoader 工作进程数
  --prefetch-factor N   数据预取因子
  --input-channels N    输入图像通道数
  --output-channels N   输出掩码通道数
  --gradient-clip V     梯度裁剪阈值
  --weight-decay V      权重衰减 (L2 正则化)
  --amp                 启用自动混合精度
  --tensorboard         启用 TensorBoard 日志
  --save-model          保存最佳模型
  --verbose             详细日志输出
  --debug               调试模式
  --device DEVICE       指定设备 (cuda/cpu)
```

#### 推荐配置

**小规模数据集 (<100 图像)**:
```bash
python train.py --epochs 30 --batch-size 8 --learning-rate 1e-3 \
  --data-limit 50 --num-workers 4 --amp
```

**中等规模 (100-500 图像)**:
```bash
python train.py --epochs 100 --batch-size 4 --learning-rate 1e-4 \
  --num-workers 6 --prefetch-factor 3 --amp --save-model
```

**大规模 (>500 图像)**:
```bash
python train.py --epochs 150 --batch-size 4 --learning-rate 1e-4 \
  --num-workers 8 --prefetch-factor 4 --gradient-clip 1.0 \
  --tensorboard --save-model --verbose
```

**调试模式**:
```bash
python train.py --epochs 5 --data-limit 10 --batch-size 2 \
  --verbose --debug --check-data
```

#### 后台训练（防止 SSH 断开）

**使用 screen（推荐）**:
```bash
# 创建会话
screen -S unet_training

# 运行训练
python train.py --epochs 100 --amp

# 分离会话：Ctrl+A+D

# 重新连接
screen -r unet_training
```

**使用 nohup**:
```bash
nohup python train.py --epochs 100 --amp > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 预测与评估

#### 单张图像预测
```bash
python predict.py -i image.jpg -o mask.png --model model.pth
```

#### 批量预测
```bash
python predict.py -i image1.jpg image2.jpg --viz --no-save
```

#### 完整参数
```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

从输入图像预测掩码

optional arguments:
  -h, --help            显示帮助信息
  --model FILE, -m FILE
                        模型文件路径
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        输入图像文件名
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        输出掩码文件名
  --viz, -v             可视化处理过程
  --no-save, -n         不保存输出掩码
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        掩码阈值 (默认：0.5)
  --scale SCALE, -s SCALE
                        输入图像缩放因子
```

#### 评估模型
```bash
python evaluate.py --model model.pth --data-dir ./data/
```

---

## 高级功能

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir runs/

# 浏览器访问 http://localhost:6006
```

**监控内容**:
- 损失曲线（训练/验证）
- Dice 系数
- 学习率变化
- 梯度直方图
- 预测掩码可视化

### Weights & Biases 集成

```bash
# 设置 API Key（可选）
export WANDB_API_KEY=your_api_key

# 启用 W&B
python train.py --amp --wandb
```

训练时会打印仪表盘链接，点击即可查看实时训练状态。

### 自动化备份

训练结束后自动生成：
- `runs/backup_YYYYMMDD_HHMMSS.tar.gz` - 包含模型、日志、配置
- `runs/models/model_best.pth` - 最佳模型
- `runs/histories/history_*.npy` - 训练历史

### 诊断工具

**快速诊断**:
```bash
bash quick_diagnostics.sh
```

**综合诊断**:
```bash
bash comprehensive_diagnostics.sh
```

**手动检查**:
```bash
# 数据流测试
python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/

# 掩码分析
python mask_analysis.py

# 维度验证
python validate_dimensions.py
```

---

## 故障排查

### 常见问题

#### 1. CUDA Out of Memory
```bash
# 解决方案：减小 batch size，增加 num-workers
python train.py --batch-size 2 --num-workers 8 --prefetch-factor 4
```

#### 2. 数据加载错误
```bash
# 检查数据格式
python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/

# 验证维度
python validate_dimensions.py
```

#### 3. 训练提前终止
检查日志：
```bash
tail -f runs/logs/training_*.log
grep -i "error\|exception" runs/logs/*.err
```

#### 4. NaN损失
```bash
# 降低学习率，启用梯度裁剪
python train.py --learning-rate 1e-5 --gradient-clip 0.5
```

### 日志查看

```bash
# 实时查看训练日志
tail -f runs/logs/training_*.log

# 查看错误日志
tail -50 runs/logs/training_*.err

# 搜索特定错误
grep -i "CUDA OOM" runs/logs/*.err
```

---

## 性能优化

### 内存优化

**激进的垃圾回收**（参考 MultiResUNet）:
```bash
# 修改 train.py 添加定期 GC
import gc
import torch

# 每个 epoch 后清理
if epoch % 3 == 0:
    torch.cuda.empty_cache()
    gc.collect()
```

**DataLoader 优化**:
```bash
# 32 核 CPU 推荐配置
python train.py --num-workers 8 --prefetch-factor 4 --pin-memory
```

### 训练加速

**混合精度训练**:
```bash
python train.py --amp  # 节省 50% 内存，提升 2-3 倍速度
```

**图像缩放**:
```bash
python train.py --scale 0.5  # 减少 75% 内存，精度损失<1%
```

### 大规模数据训练

```bash
# 5000+ 样本推荐配置
python train.py --batch-size 2 --num-workers 12 \
  --prefetch-factor 3 --gradient-clip 1.0 \
  --gradient-accumulation-steps 4
```

---

## 引用

### 原始论文

**U-Net**:
```bibtex
@article{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

### 相关资源

- [原始论文](https://arxiv.org/abs/1505.04597)
- [官方 PyTorch 实现](https://github.com/milesial/Pytorch-UNet)
- [Kaggle 竞赛页面](https://www.kaggle.com/c/carvana-image-masking-challenge)

---

## 许可证

MIT License

## 致谢

感谢原始作者 [Milesial](https://github.com/milesial) 的优秀实现。

本项目在其基础上进行了本地化增强，添加了更多实用功能和文档。
