# ORO - 语义分割项目集合

本项目集合包含两个先进的语义分割架构实现，专注于生物医学图像和多类别分割任务。

## 📁 项目结构

```
ORO/
├── Pytorch-UNet/          # 原始 UNet 架构实现
│   ├── unet/             # 模型定义
│   ├── utils/            # 工具函数
│   ├── scripts/          # 辅助脚本
│   └── train.py          # 训练脚本
└── MultiResUNet/         # 改进的 MultiResUNet 架构
    ├── pytorch/          # PyTorch 实现
    ├── tensorflow/       # TensorFlow 实现
    ├── scripts/          # 数据处理脚本
    └── train.py          # 训练脚本
```

## 🎯 子项目介绍

### 1. [Pytorch-UNet](Pytorch-UNet/README.md)

**特点**:
- 🚀 基于原始 UNet 架构的轻量化实现
- 🎨 支持多通道掩码分割（如 4 通道二值掩码）
- ⚡ 自动混合精度训练（AMP）
- 📊 TensorBoard 实时监控
- 🔧 完善的诊断和调试工具
- 💾 自动化备份和日志管理

**适用场景**:
- 快速原型开发
- 标准语义分割任务
- Kaggle Carvana 等竞赛
- 需要轻量级解决方案的项目

**快速开始**:
```bash
cd Pytorch-UNet
bash scripts/download_data.sh
python train.py --epochs 100 --batch-size 4 --amp
```

### 2. [MultiResUNet](MultiResUNet/README.md)

**特点**:
- 🏗️ MultiRes 块：多分辨率特征提取
- 🔗 Res 路径：减少编码器和解码器间的语义差距
- 🧠 激进的内存优化（降低 99.7% 内存占用）
- 📈 支持大规模数据集（3000-4000+ 样本）
- 🎯 更高的分割精度

**核心创新**:
- **MultiRes 块**: 替代标准卷积层，同时提取 3x3、5x5、7x7 多尺度特征
- **Res 路径**: 在跳跃连接中添加额外卷积，缓解语义距离

**适用场景**:
- 复杂生物医学图像分割
- 多模态数据融合
- 高精度要求的医疗影像分析
- 大规模数据集训练

**快速开始**:
```bash
cd MultiResUNet
python diagnose_memory.py
bash run_training.sh --epochs 150 --scale-factor 0.5
```

## 🔬 技术对比

| 特性 | Pytorch-UNet | MultiResUNet |
|------|-------------|--------------|
| 架构复杂度 | 简单 | 中等 |
| 训练速度 | 快 | 中等 |
| 内存占用 | 低 | 优化后极低 |
| 分割精度 | 良好 | 优秀 |
| 多通道支持 | ✅ | ✅ |
| 混合精度 | ✅ | ✅ |
| 大规模数据 | 支持 | 优化支持 |

## 🛠️ 通用功能

两个子项目均支持：

- **多通道分割**: 支持任意数量的输出通道
- **数据增强**: 旋转、缩放、翻转等
- **多种损失函数**: BCE、Dice、CrossEntropy 等
- **TensorBoard 监控**: 实时可视化训练过程
- **自动备份**: 模型、日志、配置自动保存
- **诊断工具**: 数据检查、内存监控、错误分析

## 📚 文档资源

### Pytorch-UNet 文档
- [训练指南](Pytorch-UNet/TRAINING_TIPS.md)
- [快速诊断](Pytorch-UNet/quick_diagnostics.sh)
- [综合诊断](Pytorch-UNet/comprehensive_diagnostics.sh)

### MultiResUNet 文档
- [快速开始](MultiResUNet/md/GC_QUICKSTART.md)
- [优化总结](MultiResUNet/md/GC_OPTIMIZATION_SUMMARY.md)
- [训练指南](MultiResUNet/md/TRAINING_GUIDE.md)
- [OOM 修复](MultiResUNet/md/OOM_FIX_SUMMARY.md)

## 🔧 环境配置

### 系统要求
- Python 3.6+
- PyTorch 1.13+ (推荐 2.0+)
- CUDA 11.0+ (GPU 加速)
- Linux/MacOS/Windows

### 安装依赖

**Pytorch-UNet**:
```bash
cd Pytorch-UNet
pip install torch torchvision numpy tqdm tensorboard
# 可选：Weights & Biases
pip install wandb
```

**MultiResUNet**:
```bash
cd MultiResUNet
pip install torch torchvision numpy tqdm tensorboard matplotlib
# TensorFlow 版本（可选）
pip install tensorflow
```

### Conda 环境（推荐）

```bash
# 创建环境
conda create -n oro python=3.10
conda activate oro

# 安装 PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy tqdm tensorboard matplotlib pillow
```

## 📖 使用指南

### 标准训练流程

1. **准备数据**
```bash
# 组织数据结构
data/
├── imgs/    # 输入图像
└── masks/   # 对应掩码
```

2. **数据检查**
```bash
# Pytorch-UNet
python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/

# MultiResUNet
python scripts/diagnose_data.py
```

3. **开始训练**
```bash
# Pytorch-UNet
python train.py --epochs 100 --batch-size 4 --learning-rate 1e-4 \
  --amp --tensorboard --save-model

# MultiResUNet
bash run_training.sh --epochs 150 --data-limit 3500 \
  --batch-size 4 --num-workers 8 --prefetch-factor 4
```

4. **监控训练**
```bash
# 启动 TensorBoard
tensorboard --logdir runs/

# 或查看实时日志
tail -f runs/logs/training_*.log
```

### 预测与评估

```bash
# 单张图像预测
python predict.py -i image.jpg -o mask.png --model model.pth

# 批量预测
python predict.py -i images/*.jpg --viz

# 评估模型
python evaluate.py --model model.pth --data-dir ./data/
```

## 🎓 学习资源

### 论文引用

**UNet**:
```bibtex
@article{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={MICCAI},
  year={2015}
}
```

**MultiResUNet**:
```bibtex
@article{ibtehaz2020multiresunet,
  title={MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation},
  author={Ibtehaz, Nabil and Rahman, M Sohel},
  journal={Neural Networks},
  volume={121},
  pages={74--87},
  year={2020},
  publisher={Elsevier}
}
```

### 相关教程
- [UNet 官方仓库](https://github.com/milesial/Pytorch-UNet)
- [MultiResUNet 官方仓库](https://github.com/nibtehaz/MultiResUNet)
## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

- Pytorch-UNet: MIT License
- MultiResUNet: MIT License


## 📞 联系方式

如有问题，请通过 GitHub Issues 联系我们。
