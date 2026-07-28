#!/bin/bash
set -euo pipefail

# Run from the MultiResUNet directory on the server:
#   cd ~/ORO/MultiResUNet
#   bash ../temp.sh
#
# Expected dataset layout, relative to the ORO repo root:
#   ../data/20260204111923/train/images
#   ../data/20260204111923/train/masks
#   ../data/385-liver.v1i.yolov8/train/images
#   ../data/385-liver.v1i.yolov8/train/masks
#
# Optional one-time mask conversion after re-extracting YOLO labels:
#   RUN_MASK_CONVERT=1 bash ../temp.sh

mkdir -p runs/logs

DATA_A="../data/20260204111923"
DATA_B="../data/385-liver.v1i.yolov8"

if [[ "${RUN_MASK_CONVERT:-0}" == "1" ]]; then
  echo "============================================================"
  echo "Converting YOLO labels to NPZ masks: ${DATA_A}"
  echo "============================================================"
  python scripts/yolo_to_npz.py \
    --data-root "${DATA_A}" \
    --splits train valid test \
    --num-classes 4

  echo "============================================================"
  echo "Converting YOLO labels to NPZ masks: ${DATA_B}"
  echo "============================================================"
  python scripts/yolo_to_npz.py \
    --data-root "${DATA_B}" \
    --splits train valid test \
    --num-classes 4
fi

COMMON_ARGS=(
  --split-mode fixed
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
  local data_root="$2"
  shift 2
  local log_file="runs/logs/run_${name}.log"

  echo "============================================================"
  echo "Starting ${name}"
  echo "Dataset: ${data_root}"
  echo "Log: ${log_file}"
  echo "============================================================"

  python train.py "${COMMON_ARGS[@]}" "$@" \
    --train-img-dir "${data_root}/train/images" \
    --train-mask-dir "${data_root}/train/masks" \
    --val-img-dir "${data_root}/valid/images" \
    --val-mask-dir "${data_root}/valid/masks" \
    --test-img-dir "${data_root}/test/images" \
    --test-mask-dir "${data_root}/test/masks" \
    --save-dir "models/${name}" \
    --log-dir "runs/logs/${name}" \
    > "${log_file}" 2>&1

  echo "Finished ${name}"
  echo
}

# A source: 20260204111923. Larger valid/test split, better for judging generalization.
run_exp "U_A_20260204_anchor_scale075_cls2w10" "${DATA_A}" \
  --class-weights 1 1 1 1

# B source: 385-liver.v1i.yolov8. Smaller valid/test split, mainly checks source-specific behavior.
run_exp "U_B_385liver_anchor_scale075_cls2w10" "${DATA_B}" \
  --class-weights 1 1 1 1

# A full-resolution probe. If A improves clearly, resolution is useful on the cleaner/larger source.
run_exp "U_A_20260204_fullres_cls2w10" "${DATA_A}" \
  --scale-factor 1.0 \
  --batch-size 8 \
  --class-weights 1 1 1 1

# B full-resolution probe. This is risky because B test is small, but it checks whether detail helps this source.
run_exp "U_B_385liver_fullres_cls2w10" "${DATA_B}" \
  --scale-factor 1.0 \
  --batch-size 8 \
  --class-weights 1 1 1 1

echo "Separate-source fixed train/valid/test runs completed."
