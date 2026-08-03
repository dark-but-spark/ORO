#!/bin/bash
set -uo pipefail

# Run on the server from MultiResUNet:
#   cd ~/ORO/MultiResUNet
#   bash ../temp_B_high.sh
#
# Purpose:
#   B-only high-score pilots. Train on original B train, validate on curated B
#   valid, and report on curated B test. Curated valid/test are not used for
#   training.
#
# Default plan:
#   Run 4 B-only pilots sequentially. Do not start many jobs at once.
#
# Optional expansion:
#   RUN_B_EXTRA=1 bash ../temp_B_high.sh

mkdir -p runs/logs

DATA_B="../data/385-liver.groupclean.v1"
DATA_B_CURATED="../data/385-liver.groupclean.v1_curated_eval_20260802"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_B_EXTRA="${RUN_B_EXTRA:-0}"

FAILED_TASKS=()

COMMON_ARGS=(
  --split-mode fixed
  --input-channels 3
  --output-channels 4
  --epochs 180
  --learning-rate 2e-5
  --gradient-clip 0.5
  --weight-decay 2e-4
  --num-workers 4
  --prefetch-factor 2
  --repeat-factor 1
  --train-augmentation
  --augmentation-strength mild
  --augmentation-curriculum none
  --use-combined-loss
  --bce-weight 0.7
  --dice-weight 0.3
  --class-weights 1 1 1 1
  --model-architecture smp_unet
  --encoder-weights imagenet
  --dropout-rate 0.2
  --lr-scheduler cosine
  --early-stopping-min-epochs 80
  --early-stopping-patience 35
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
)

validate_dirs() {
  for path in \
    "${DATA_B}/train/images" \
    "${DATA_B}/train/masks" \
    "${DATA_B_CURATED}/valid/images" \
    "${DATA_B_CURATED}/valid/masks" \
    "${DATA_B_CURATED}/test/images" \
    "${DATA_B_CURATED}/test/masks"; do
    if [[ ! -d "${path}" ]]; then
      echo "Missing directory: ${path}" >&2
      return 1
    fi
  done
}

run_train() {
  local name="$1"
  local seed="$2"
  local batch_size="$3"
  shift 3

  local log_file="runs/logs/run_${name}.log"

  echo "============================================================"
  echo "Starting ${name}"
  echo "Seed: ${seed}"
  echo "Batch size: ${batch_size}"
  echo "Log: ${log_file}"
  echo "============================================================"

  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  if python train.py "${COMMON_ARGS[@]}" \
    --train-img-dir "${DATA_B}/train/images" \
    --train-mask-dir "${DATA_B}/train/masks" \
    --val-img-dir "${DATA_B_CURATED}/valid/images" \
    --val-mask-dir "${DATA_B_CURATED}/valid/masks" \
    --test-img-dir "${DATA_B_CURATED}/test/images" \
    --test-mask-dir "${DATA_B_CURATED}/test/masks" \
    --batch-size "${batch_size}" \
    --seed "${seed}" \
    --save-dir "models/${name}" \
    --log-dir "runs/logs/${name}" \
    "$@" \
    > "${log_file}" 2>&1; then
    echo "Finished ${name}"
  else
    local rc=$?
    echo "FAILED ${name} (exit ${rc}); continuing with the next run." >&2
    FAILED_TASKS+=("${name}:exit_${rc}")
  fi
}

if ! validate_dirs; then
  echo "Dataset check failed. Sync curated B dataset to the server first:" >&2
  echo "  ../data/385-liver.groupclean.v1_curated_eval_20260802" >&2
  exit 1
fi

# Stage 1: four sequential pilots.
run_train "Bhigh_resnet34_fullres_seed42" 42 8 \
  --encoder-name resnet34

run_train "Bhigh_resnet34_scale075_seed42" 42 16 \
  --scale --scale-factor 0.75 \
  --encoder-name resnet34

run_train "Bhigh_resnet34_fullres_dropout03_seed42" 42 8 \
  --encoder-name resnet34 \
  --dropout-rate 0.3

run_train "Bhigh_resnet34_fullres_plateau_seed42" 42 8 \
  --encoder-name resnet34 \
  --lr-scheduler plateau \
  --lr-patience 8

# Stage 2: enable only after Stage 1 indicates which direction is promising.
if [[ "${RUN_B_EXTRA}" == "1" ]]; then
  run_train "Bhigh_effb3_fullres_seed42" 42 6 \
    --encoder-name efficientnet-b3

  run_train "Bhigh_resnet34_fullres_seed43" 43 8 \
    --encoder-name resnet34

  run_train "Bhigh_resnet34_fullres_seed44" 44 8 \
    --encoder-name resnet34
else
  echo "Skipping B extra pilots (set RUN_B_EXTRA=1 to enable)."
fi

echo "============================================================"
if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "Completed with ${#FAILED_TASKS[@]} failed task(s):"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  echo "Inspect runs/logs/run_Bhigh_*.log"
  exit 1
fi

echo "All B-high pilots completed successfully."
echo "Training logs: runs/logs/run_Bhigh_*.log"
echo "Run directories: runs/Bhigh_*"
