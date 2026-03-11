#!/bin/bash
# Memory Monitor for MultiResUNet Training
# Usage: ./monitor_memory.sh [training_pid]

echo "=========================================="
echo "  MultiResUNet Memory Monitor"
echo "=========================================="
echo ""

# If PID provided, monitor specific process
if [ -n "$1" ]; then
    TRAIN_PID=$1
    echo "Monitoring training process PID: $TRAIN_PID"
else
    # Find Python training process
    TRAIN_PID=$(pgrep -f "python.*train.py" | head -1)
    if [ -n "$TRAIN_PID" ]; then
        echo "Found training process PID: $TRAIN_PID"
    else
        echo "No training process found. Run in background..."
    fi
fi

echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Main monitoring loop
while true; do
    TIMESTAMP=$(date '+%H:%M:%S')
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$TIMESTAMP] System Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # System memory
    echo "📊 System Memory:"
   free -h | grep Mem | awk '{printf "  Total: %s | Used: %s | Free: %s\n", $2, $3, $4}'
    
    # Swap usage
   free -h | grep Swap | awk '{printf "  Swap: %s | Used: %s | Free: %s\n", $2, $3, $4}'
    
    # GPU memory (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "🎮 GPU Status:"
       nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory \
                   --format=csv,noheader,nounits 2>/dev/null | while read line; do
            echo "  GPU $line"
        done
    fi
    
    # Process-specific memory
    if [ -n "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo ""
        echo "🔍 Training Process (PID: $TRAIN_PID):"
        
        # RSS memory in human-readable format
        RSS_KB=$(ps -o rss= -p $TRAIN_PID 2>/dev/null)
        if [ -n "$RSS_KB" ]; then
            RSS_MB=$((RSS_KB / 1024))
            if [ $RSS_MB -gt 1024 ]; then
                RSS_GB=$((RSS_MB / 1024))
                echo "  RSS Memory: ${RSS_GB} GB"
            else
                echo "  RSS Memory: ${RSS_MB} MB"
            fi
        fi
        
        # CPU usage
        CPU_PCT=$(ps -o %cpu= -p $TRAIN_PID 2>/dev/null)
        if [ -n "$CPU_PCT" ]; then
            echo "  CPU Usage: ${CPU_PCT}%"
        fi
        
        # Check if process is still running
        STATE=$(ps -o state= -p $TRAIN_PID 2>/dev/null)
        case $STATE in
            R*) STATUS="Running" ;;
            S*) STATUS="Sleeping" ;;
            D*) STATUS="Disk Sleep" ;;
            Z*) STATUS="Zombie" ;;
            *) STATUS="Unknown" ;;
        esac
        echo "  State: $STATUS"
    else
        if [ -n "$TRAIN_PID" ]; then
            echo "  ⚠ Training process not found!"
            # Try to find a new one
            TRAIN_PID=$(pgrep -f "python.*train.py" | head -1)
            if [ -n "$TRAIN_PID" ]; then
                echo "  ✓ Found new training process: $TRAIN_PID"
            fi
        fi
    fi
    
    # Top 5 memory-consuming processes
    echo ""
    echo "📈 Top 5 Memory Processes:"
    ps aux --sort=-%mem | head -6 | tail -5 | \
        awk '{printf "  %-8s %6s%% %7s %s\n", $2, $4, $6, $11}'
    
    # OOM check
    if dmesg 2>/dev/null | grep -i "killed process" | tail -1 | grep -q python; then
        echo ""
        echo "⚠️  WARNING: Recent OOM kill detected!"
       dmesg 2>/dev/null | grep -i "killed process" | grep python | tail -1
    fi
    
    # Wait before next update
    sleep 10
done
