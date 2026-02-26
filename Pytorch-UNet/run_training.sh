#!/bin/bash

# Activate the Python environment
source /share/home/zjm/anaconda3/bin/activate zjm

# Navigate to the project directory
cd /share/home/zjm/ORO/Pytorch-UNet

# Run the training script
python train.py \
    --epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --scale 0.5 \
    --validation 10 \
    --classes 4 \
    --mask-channels 4