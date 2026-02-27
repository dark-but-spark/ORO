#!/bin/bash

# Activate the Python environment
source /share/home/zjm/anaconda3/bin/activate zjm

# Navigate to the project directory
cd /share/home/zjm/ORO/Pytorch-UNet

# Create a timestamped log file
LOG_FILE="runs/training_$(date +%Y%m%d_%H%M%S).log"

# Run the training script and log output
python train.py \
    --epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --scale 0.5 \
    --validation 10 \
    --classes 4 \
    --mask-channels 4 &> "$LOG_FILE"

# Compress the runs directory
BACKUP_FILE="runs_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" -C runs .

# Delete the contents of the runs directory
rm -rf runs/*