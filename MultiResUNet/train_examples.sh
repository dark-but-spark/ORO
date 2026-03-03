#!/bin/bash
# MultiResUNet 训练测试脚本示例 - Linux 版本
# 使用新的 run_training.sh 脚本

# Activate the Python environment
source /share/home/zjm/anaconda3/bin/activate zjm

# Navigate to the project directory
cd /share/home/zjm/ORO/Pytorch-UNet


set -e  # Exit on error

echo "=========================================="
echo "MultiResUNet 训练命令示例 (Linux)"
echo "=========================================="
echo ""

# 设置项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. 快速测试（仅 5 个样本，3 个 epoch）
echo "1. 快速测试配置（5 个样本，3 个 epoch）..."
echo "命令：bash run_training.sh --epochs 3 --data-limit 5 --batch-size 2"
echo ""
bash run_training.sh --epochs 3 --data-limit 5 --batch-size 2

# 2. 调试模式（验证数据 + 详细日志）
echo "2. 调试模式（验证数据 + 详细日志）..."
echo "命令：bash run_training.sh --epochs 5 --data-limit 10 --debug --verbose"
echo ""
bash run_training.sh --epochs 5 --data-limit 10 --debug --verbose

# 3. 保存最佳模型
echo "3. 保存最佳模型..."
echo "命令：bash run_training.sh --epochs 10 --data-limit 20 --save-model"
echo ""
bash run_training.sh --epochs 10 --data-limit 20 --save-model

# 4. 学习率测试
echo "4. 测试不同学习率..."
echo "命令：bash run_training.sh --epochs 10 --data-limit 20 --learning-rate 1e-3"
echo ""
bash run_training.sh --epochs 10 --data-limit 20 --learning-rate 1e-3

# 5. 梯度裁剪测试
echo "5. 梯度裁剪测试..."
echo "命令：bash run_training.sh --epochs 10 --data-limit 20 --gradient-clip 0.5 --debug"
echo ""
bash run_training.sh --epochs 10 --data-limit 20 --gradient-clip 0.5 --debug

# 6. 完整训练配置
echo "6. 完整训练配置..."
echo "命令：bash run_training.sh --epochs 50 --data-limit 100 --batch-size 4 --save-model"
echo ""
bash run_training.sh --epochs 50 --data-limit 100 --batch-size 4 --save-model

# 7. 批量实验示例
echo "7. 批量实验示例（测试不同学习率）..."
echo "命令：依次运行 3 个不同学习率的实验"
echo ""
echo "实验 1: 学习率 1e-3"
bash run_training.sh --epochs 30 --data-limit 100 --learning-rate 1e-3

echo "实验 2: 学习率 1e-4"
bash run_training.sh --epochs 30 --data-limit 100 --learning-rate 1e-4

echo "实验 3: 学习率 5e-5"
bash run_training.sh --epochs 30 --data-limit 100 --learning-rate 5e-5

# 8. 查看训练结果
echo "8. 查看训练结果..."
echo "命令：查看最近的训练日志和备份"
echo ""
echo "查看最近的日志："
echo "  ls -lt runs/logs/ | head -5"
echo ""
echo "查看最近的备份："
echo "  ls -lt runs/backup_*.tar.gz | head -5"
echo ""
echo "查看最近的配置清单："
echo "  ls -lt runs/manifest_*.txt | head -5"
echo ""

# 9. 对比实验结果
echo "9. 对比实验结果..."
echo "命令：快速对比所有实验的 Dice 系数"
echo ""
echo "提取所有日志中的最终 Dice 系数："
echo "  grep 'Final Dice' runs/logs/training_*.log 2>/dev/null || echo '暂无训练日志'"
echo ""
grep 'Final Dice' runs/logs/training_*.log 2>/dev/null || echo '暂无训练日志'
echo "提取所有日志中的最佳 Dice 系数："
echo "  grep 'Best Validation Dice' runs/logs/training_*.log 2>/dev/null || echo '暂无训练日志'"
echo ""
grep 'Best Validation Dice' runs/logs/training_*.log 2>/dev/null || echo '暂无训练日志'

# 10. 分析训练历史
echo "10. 分析训练历史..."
echo "命令：使用 Python 分析训练历史"
echo ""
cat << 'PYTHON_SCRIPT'
# 保存为 analyze_history.py 并运行
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 查找所有历史文件
history_files = sorted(glob.glob('runs/histories/history_*.npy'))

if not history_files:
    print("未找到训练历史文件")
else:
    print(f"找到 {len(history_files)} 个历史文件:")
    for hf in history_files[-5:]:  # 显示最近 5 个
        print(f"  - {hf}")
    
    # 加载最近的历史
    latest = history_files[-1]
    history = np.load(latest, allow_pickle=True).item()
    
    print(f"\n加载最新历史：{latest}")
    print(f"  训练轮次：{len(history['train_loss'])}")
    print(f"  最终 Dice: {history['val_dice'][-1]:.4f}")
    print(f"  最终 Jaccard: {history['val_jaccard'][-1]:.4f}")
    print(f"  最佳 Dice: {max(history['val_dice']):.4f}")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training and Validation Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Dice 系数
    axes[1].plot(history['val_dice'], label='Dice', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].set_title('Validation Dice Coefficient')
    axes[1].grid(True, alpha=0.3)
    
    # Jaccard 指数
    axes[2].plot(history['val_jaccard'], label='Jaccard', color='orange', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Jaccard Index')
    axes[2].legend()
    axes[2].set_title('Validation Jaccard Index')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'training_curve_{os.path.basename(latest).replace(".npy", ".png")}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存：{save_path}")
    plt.show()
PYTHON_SCRIPT
echo ""

echo "=========================================="
echo "快速开始命令"
echo "=========================================="
echo ""
echo "# 1. 快速测试（推荐首次使用）"
echo "bash run_training.sh --epochs 5 --data-limit 10"
echo ""
echo "# 2. 调试模式"
echo "bash run_training.sh --epochs 10 --data-limit 20 --debug --verbose"
echo ""
echo "# 3. 正式训练"
echo "bash run_training.sh --epochs 50 --data-limit 200 --batch-size 4 --save-model"
echo ""
echo "# 4. 查看帮助"
echo "bash run_training.sh --help"
echo ""

echo "=========================================="
echo "文件位置"
echo "=========================================="
echo ""
echo "训练日志：runs/logs/training_*.log"
echo "模型文件：runs/models/"
echo "训练历史：runs/histories/history_*.npy"
echo "备份文件：runs/backup_*.tar.gz"
echo "配置清单：runs/manifest_*.txt"
echo ""

echo "=========================================="
echo "使用说明"
echo "=========================================="
echo ""
echo "1. 取消注释上面的命令以运行特定测试"
echo "2. 或直接使用快速开始命令"
echo "3. 所有结果自动保存在 runs/ 目录"
echo "4. 使用 grep 对比不同实验的结果"
echo ""
echo "示例：对比所有实验的 Dice 系数"
echo "  grep 'Final Dice' runs/logs/training_*.log"
echo ""
echo "示例：查看最近的备份"
echo "  ls -lt runs/backup_*.tar.gz | head -5"
echo ""
