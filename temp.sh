#!/bin/bash
set -euo pipefail

# Run from the MultiResUNet directory on the server:
#   cd ~/ORO/MultiResUNet
#   bash ../temp.sh
#
# Optional one-time mask conversion after re-extracting YOLO labels:
#   RUN_MASK_CONVERT=1 bash ../temp.sh
#
# Evaluation policy:
#   train: data/train
#   val:   data/valid, used for early stopping and best checkpoint selection
#   test:  data/test, evaluated once after training

mkdir -p runs/logs

if [[ "${RUN_MASK_CONVERT:-0}" == "1" ]]; then
  echo "============================================================"
  echo "Converting YOLO labels to NPZ masks for train/valid/test"
  echo "============================================================"
  python scripts/yolo_to_npz.py \
    --data-root data \
    --splits train valid test \
    --num-classes 4
fi

COMMON_ARGS=(
  --split-mode fixed
  --train-img-dir data/train/images
  --train-mask-dir data/train/masks
  --val-img-dir data/valid/images
  --val-mask-dir data/valid/masks
  --test-img-dir data/test/images
  --test-mask-dir data/test/masks
  --scale
  --scale-factor 0.75
  --input-channels 3
  --output-channels 4
  --model-architecture smp_unet
  --encoder-name resnet34
  --encoder-weights imagenet
  --epochs 140
  --batch-size 16
  --learning-rate 2e-5
  --gradient-clip 0.5
  --weight-decay 2e-4
  --num-workers 4
  --prefetch-factor 2
  --repeat-factor 1
  --train-augmentation
  --augmentation-strength mild
  --use-combined-loss
  --bce-weight 0.7
  --dice-weight 0.3
  --lr-scheduler cosine
  --early-stopping-min-epochs 80
  --early-stopping-patience 25
  --checkpoint-interval 0
  --tensorboard
  --tb-image-interval 0
  --tb-num-images 0
  --verbose
  --save-model
  --val-tta none
  --test-tta none
  --test-threshold 0.5
  --metric-ignore-classes 2
  --device cuda
  --seed 42
)

run_exp() {
  local name="$1"
  shift
  local log_file="runs/logs/run_${name}.log"

  echo "============================================================"
  echo "Starting ${name}"
  echo "Log: ${log_file}"
  echo "============================================================"

  python train.py "${COMMON_ARGS[@]}" "$@" \
    --save-dir "models/${name}" \
    --log-dir "runs/logs/${name}" \
    > "${log_file}" 2>&1

  echo "Finished ${name}"
  echo
}

# Current fixed-split winner. Rerun as the new anchor only if you need a clean
# comparison in the same batch.
run_exp "S_fixed_cls2w05_anchor" \
  --class-weights 1 1 0.5 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1

# Check whether class2 oversampling is worsening overfit on the fixed valid/test.
run_exp "S_fixed_cls2w05_no_os" \
  --class-weights 1 1 0.5 1

# Full resolution probe. Use smaller batch to stay within memory.
run_exp "S_fixed_cls2w05_scale10" \
  --scale-factor 1.0 \
  --batch-size 8 \
  --class-weights 1 1 0.5 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1

# Stronger regularization for the large train/valid gap.
run_exp "S_fixed_cls2w05_reg" \
  --dropout-rate 0.4 \
  --weight-decay 5e-4 \
  --class-weights 1 1 0.5 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1

# Mild -> moderate curriculum may improve fixed-test robustness without jumping to strong aug.
run_exp "S_fixed_cls2w05_curr_l04" \
  --augmentation-curriculum cosine \
  --curriculum-start-epoch 30 \
  --curriculum-ramp-epochs 35 \
  --curriculum-max-aug-level 0.4 \
  --curriculum-target-strength moderate \
  --class-weights 1 1 0.5 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1

echo "Fixed train/valid/test diagnostic training completed."
