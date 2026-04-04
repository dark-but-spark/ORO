#!/bin/bash
# 示例脚本：展示不同学习率调度器的使用方法

echo "MultiResUNet 学习率调度器示例"
echo "=============================="

# 示例1：余弦退火调度器（默认）
echo "示例1：余弦退火调度器（默认）"
echo "特点：平滑地降低学习率，适合大多数情况"
echo "命令："
echo "python train.py --lr-scheduler cosine --learning-rate 1e-4"
echo ""

# 示例2：步长调度器
echo "示例2：步长调度器"
echo "特点：每隔固定周期将学习率乘以gamma因子"
echo "命令："
echo "python train.py --lr-scheduler step --lr-step-size 30 --lr-gamma 0.1 --learning-rate 1e-4"
echo ""

# 示例3：指数调度器
echo "示例3：指数调度器"
echo "特点：每个epoch后按指数递减学习率"
echo "命令："
echo "python train.py --lr-scheduler exponential --lr-gamma 0.95 --learning-rate 1e-4"
echo ""

# 示例4：自适应调度器（ReduceLROnPlateau）
echo "示例4：自适应调度器（ReduceLROnPlateau）"
echo "特点：当监控指标停止改善时降低学习率，最适合防止过拟合"
echo "命令："
echo "python train.py --lr-scheduler plateau --lr-patience 10 --lr-gamma 0.5 --learning-rate 1e-4"
echo ""

# 推荐用于过拟合场景的配置
echo "推荐用于过拟合场景的配置："
echo "python train.py \\"
echo "  --epochs 150 \\"
echo "  --batch-size 4 \\"
echo "  --learning-rate 5e-5 \\"
echo "  --weight-decay 1e-4 \\"
echo "  --dropout-rate 0.3 \\"
echo "  --lr-scheduler plateau \\"
echo "  --lr-patience 10 \\"
echo "  --lr-gamma 0.5 \\"
echo "  --use-combined-loss \\"
echo "  --dice-weight 0.6 \\"
echo "  --bce-weight 0.4 \\"
echo "  --tensorboard \\"
echo "  --save-model"