#!/bin/bash
# Example script to address overfitting in MultiResUNet training
# This script implements the recommendations from the analysis

echo "MultiResUNet Training with Overfitting Prevention"
echo "==============================================="

# Set basic training parameters
EPOCHS=200
BATCH_SIZE=4
LEARNING_RATE=5e-5
WEIGHT_DECAY=1e-4

# Regularization parameters to prevent overfitting
DROPOUT_RATE=0.3
GRADIENT_CLIP=0.5

echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Dropout rate: $DROPOUT_RATE"
echo ""

# Generate timestamp for unique log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting training with overfitting prevention measures..."
echo ""

# Run training with enhanced regularization and early stopping
python train.py \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --dropout-rate $DROPOUT_RATE \
  --gradient-clip $GRADIENT_CLIP \
  --data-limit 3500 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --save-model \
  --tensorboard \
  --use-combined-loss \
  --bce-weight 0.4 \
  --dice-weight 0.6 \
  --verbose

echo ""
echo "Training completed. Check TensorBoard logs for results:"
echo "  tensorboard --logdir runs/tensorboard"