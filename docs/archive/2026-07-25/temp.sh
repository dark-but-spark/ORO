#!/bin/bash
set -euo pipefail

# Run from the MultiResUNet directory on the server:
#   cd ~/zjm/ORO1/ORO/MultiResUNet
#   bash ../temp.sh
#
# Purpose:
#   Small controlled training probes for the class2-label-noise hypothesis.
#   Keep the current strongest SMP-UNet setup fixed, then reduce class2 loss weight
#   and compare whether class0/1/3 plus overall validation Dice improve.
#
# Space control:
#   --checkpoint-interval 0 saves only best_model.pth and final model.pth
#   --tb-image-interval 0 disables TensorBoard image panels

mkdir -p runs/logs

COMMON_ARGS=(
  --validation-split 0.1
  --scale
  --scale-factor 0.75
  --input-channels 3
  --output-channels 4
  --model-architecture smp_unet
  --encoder-name resnet34
  --encoder-weights imagenet
  --epochs 120
  --batch-size 16
  --learning-rate 2e-5
  --gradient-clip 0.5
  --weight-decay 5e-4
  --num-workers 4
  --prefetch-factor 2
  --repeat-factor 1
  --train-augmentation
  --augmentation-strength mild
  --augmentation-curriculum cosine
  --curriculum-start-epoch 30
  --curriculum-ramp-epochs 30
  --curriculum-max-aug-level 0.4
  --curriculum-target-strength moderate
  --use-combined-loss
  --bce-weight 0.7
  --dice-weight 0.3
  --lr-scheduler cosine
  --lr-cosine-t-max 100
  --early-stopping-min-epochs 70
  --early-stopping-patience 25
  --checkpoint-interval 0
  --tensorboard
  --tb-image-interval 0
  --tb-num-images 0
  --verbose
  --save-model
  --device cuda
  --seed 42
  --metric-ignore-classes 2
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

# Baseline rerun under the same script shape. This anchors this batch of probes.
run_exp "Q_cls2w125_os15_tta_anchor" \
  --class-weights 1 1 1.25 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1 \
  --val-tta flips

# Test whether reducing noisy class2 supervision releases shared representation quality.
run_exp "Q_cls2w10_os15_tta" \
  --class-weights 1 1 1.0 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1 \
  --val-tta flips

run_exp "Q_cls2w075_os15_tta" \
  --class-weights 1 1 0.75 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1 \
  --val-tta flips

run_exp "Q_cls2w05_os15_tta" \
  --class-weights 1 1 0.5 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1 \
  --val-tta flips

# Separate loss-weight effect from oversampling effect.
run_exp "Q_cls2w075_os12_tta" \
  --class-weights 1 1 0.75 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.2 \
  --oversample-min-pixels 1 \
  --val-tta flips

run_exp "Q_cls2w10_noos_tta" \
  --class-weights 1 1 1.0 1 \
  --val-tta flips

# Check whether TTA is masking or amplifying the class2 issue during model selection.
run_exp "Q_cls2w075_os15_no_tta" \
  --class-weights 1 1 0.75 1 \
  --oversample-class-indices 2 \
  --oversample-factor 1.5 \
  --oversample-min-pixels 1 \
  --val-tta none

echo "All class2 weight probes completed."
