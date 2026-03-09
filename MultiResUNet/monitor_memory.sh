#!/bin/bash
echo "=== MultiResUNet Memory Monitor ==="
echo "Monitoring every 30 seconds..."
echo ""

while true; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')"
    
    # 系统内存
    echo "System Memory:"
    free -h | grep Mem | awk '{print "  Total: "$2", Used: "$3", Free: "$4}'
    
    # GPU 内存
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory:"
        nvidia-smi --query-gpu=memory.used,memory.free \
                   --format=csv,noheader,nounits | \
                   awk '{print "  Used: "$1" MB, Free: "$3" MB"}'
    fi
    
    # Python 进程
    echo "Python Process:"
    ps aux | grep "python.*train.py" | grep -v grep | \
        awk '{printf "  PID: %s, RSS: %.1f MB\n", $2, $6/1024}'
    
    sleep 30
done