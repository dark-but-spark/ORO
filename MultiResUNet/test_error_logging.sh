#!/bin/bash
# 错误日志捕获测试脚本

echo "=========================================="
echo "错误日志捕获功能演示"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_DIR="test_error_logs"
mkdir -p "$TEST_DIR"

echo "1️⃣ 测试：正常重定向（推荐方式）"
echo "命令：bash run_training.sh --epochs 5 --data-limit 10 --debug"
echo "输出："
echo "   - runs/logs/training_${TIMESTAMP}.log (标准输出)"
echo "   - runs/logs/training_${TIMESTAMP}.err (错误输出)"
echo ""

echo "2️⃣ 测试：手动重定向到单独文件"
echo "命令：bash run_training.sh --epochs 5 > output.out 2> error.err"
echo ""

echo "3️⃣ 测试：合并到一个文件"
echo "命令：bash run_training.sh --epochs 5 > all_output.log 2>&1"
echo ""

echo "4️⃣ 测试：后台运行 + 错误捕获"
echo "命令：nohup bash run_training.sh --epochs 5 > training.log 2>&1 &"
echo ""

echo "=========================================="
echo "查看错误的快捷命令："
echo "=========================================="
echo "tail -f runs/logs/*.err              # 实时监控错误"
echo "tail -50 runs/logs/*.err             # 查看最近 50 行错误"
echo "grep -i 'error' runs/logs/*.err      # 搜索错误"
echo "grep -A 20 'Traceback' runs/logs/*.err  # 查看异常堆栈"
echo ""

echo "📖 详细文档请查看：ERROR_LOG_GUIDE.md"
echo ""
