#!/bin/bash
set -euo pipefail

# Run from the MultiResUNet directory on the server:
#   cd ~/ORO/MultiResUNet
#   bash ../temp.sh
#
# Purpose:
#   1. Create B_clean if needed, using group-safe Roboflow splitting.
#   2. Train one B_clean anchor model.
#   3. Evaluate B_clean model on A test.
#   4. Evaluate existing A models on B_clean test for cross-source comparison.

mkdir -p runs/logs runs/debug_eval

DATA_A="../data/20260204111923"
DATA_B_RAW="../data/385-liver.v1i.yolov8"
DATA_B_CLEAN="../data/385-liver.groupclean.v1"

B_CLEAN_NAME="V_Bclean_anchor_scale075_cls2w10"

ensure_b_clean() {
  if [[ -d "${DATA_B_CLEAN}/train/images" && -d "${DATA_B_CLEAN}/train/masks" ]]; then
    echo "B_clean dataset already exists: ${DATA_B_CLEAN}"
    return
  fi

  echo "============================================================"
  echo "Creating leakage-safe B_clean dataset"
  echo "Source: ${DATA_B_RAW}"
  echo "Output: ${DATA_B_CLEAN}"
  echo "============================================================"

  python scripts/group_split_roboflow_dataset.py \
    --source-root "${DATA_B_RAW}" \
    --output-root "${DATA_B_CLEAN}" \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --max-variants-per-group 3 \
    --copy-mode hardlink \
    --convert-masks \
    --num-classes 4
}

COMMON_TRAIN_ARGS=(
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

train_b_clean() {
  local name="${B_CLEAN_NAME}"
  local log_file="runs/logs/run_${name}.log"

  echo "============================================================"
  echo "Training ${name}"
  echo "Dataset: ${DATA_B_CLEAN}"
  echo "Log: ${log_file}"
  echo "============================================================"

  python train.py "${COMMON_TRAIN_ARGS[@]}" \
    --train-img-dir "${DATA_B_CLEAN}/train/images" \
    --train-mask-dir "${DATA_B_CLEAN}/train/masks" \
    --val-img-dir "${DATA_B_CLEAN}/valid/images" \
    --val-mask-dir "${DATA_B_CLEAN}/valid/masks" \
    --test-img-dir "${DATA_B_CLEAN}/test/images" \
    --test-mask-dir "${DATA_B_CLEAN}/test/masks" \
    --class-weights 1 1 1 1 \
    --save-dir "models/${name}" \
    --log-dir "runs/logs/${name}" \
    > "${log_file}" 2>&1

  echo "Finished ${name}"
}

collect_runs() {
  local pattern="$1"
  mapfile -t FOUND_RUNS < <(find runs -maxdepth 1 -type d -name "${pattern}" | sort)
  if [[ "${#FOUND_RUNS[@]}" -eq 0 ]]; then
    echo "ERROR: no runs matched pattern: ${pattern}" >&2
    exit 1
  fi
}

evaluate_group() {
  local label="$1"
  local test_root="$2"
  shift 2
  local output_prefix="runs/debug_eval/${label}"

  echo "============================================================"
  echo "Evaluating ${label}"
  echo "Test root: ${test_root}"
  echo "Run roots: $*"
  echo "============================================================"

  python scripts/evaluate_history_on_test.py \
    --run-roots "$@" \
    --test-img-dir "${test_root}/test/images" \
    --test-mask-dir "${test_root}/test/masks" \
    --output-csv "${output_prefix}.csv" \
    --output-json "${output_prefix}.json" \
    --device cuda \
    --batch-size 8 \
    --num-workers 4 \
    --threshold 0.5 \
    --tta none \
    --metric-ignore-classes 2 \
    --default-encoder-weights none
}

ensure_b_clean
train_b_clean

collect_runs "${B_CLEAN_NAME}_*"
B_CLEAN_RUNS=("${FOUND_RUNS[@]}")

evaluate_group "own_Bclean_model_on_Bclean_test" "${DATA_B_CLEAN}" "${B_CLEAN_RUNS[@]}"
evaluate_group "cross_Bclean_model_on_A_test" "${DATA_A}" "${B_CLEAN_RUNS[@]}"

if find runs -maxdepth 1 -type d -name "U_A_*" | grep -q .; then
  collect_runs "U_A_*"
  A_RUNS=("${FOUND_RUNS[@]}")
  evaluate_group "cross_A_models_on_Bclean_test" "${DATA_B_CLEAN}" "${A_RUNS[@]}"
else
  echo "No U_A_* runs found. Skipping A models on B_clean test."
fi

echo "B_clean training and cross-source evaluation completed."
echo "Outputs:"
echo "  Training log: runs/logs/run_${B_CLEAN_NAME}.log"
echo "  Eval CSV/JSON: runs/debug_eval"
