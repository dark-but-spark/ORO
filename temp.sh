#!/bin/bash
set -uo pipefail

# Run on the server from MultiResUNet:
#   cd ~/ORO/MultiResUNet
#   bash ../temp.sh
#
# Purpose:
#   B-only parameter exploration for the final model.
#   - Target metric: 4-class global Dice. No class is ignored.
#   - Train root: B group-clean train split.
#   - Primary validation/test: B curated eval split.
#   - Original B evaluation is reference only because the raw split had leakage.
#
# Optional expansion after default runs finish:
#   RUN_EXTRA=1 bash ../temp.sh
#
# Optional reference-only evaluation on raw B:
#   RUN_REFERENCE_EVAL=1 bash ../temp.sh

mkdir -p runs/logs runs/debug_eval

DATA_B="../data/385-liver.groupclean.v1"
DATA_B_CURATED="../data/385-liver.groupclean.v1_curated_eval_20260802"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_B="${RUN_B:-1}"
RUN_EXTRA="${RUN_EXTRA:-0}"
RUN_REFERENCE_EVAL="${RUN_REFERENCE_EVAL:-1}"

FAILED_TASKS=()

COMMON_ARGS=(
  --split-mode fixed
  --input-channels 3
  --output-channels 4
  --epochs 180
  --learning-rate 2e-5
  --gradient-clip 0.5
  --weight-decay 5e-4
  --num-workers 4
  --prefetch-factor 2
  --repeat-factor 1
  --train-augmentation
  --augmentation-strength mild
  --augmentation-curriculum cosine
  --curriculum-max-aug-level 0.4
  --use-combined-loss
  --bce-weight 0.7
  --dice-weight 0.3
  --model-architecture smp_unet
  --encoder-name resnet34
  --encoder-weights imagenet
  --dropout-rate 0.2
  --lr-scheduler cosine
  --lr-cosine-t-max 100
  --early-stopping-min-epochs 90
  --early-stopping-patience 35
  --checkpoint-interval 0
  --tensorboard
  --tb-image-interval 0
  --tb-num-images 0
  --verbose
  --save-model
  --val-tta flips
  --test-tta flips
  --test-threshold 0.5
  --device cuda
)

require_split_dirs() {
  local root="$1"
  local split="$2"
  for kind in images masks; do
    if [[ ! -d "${root}/${split}/${kind}" ]]; then
      echo "Missing directory: ${root}/${split}/${kind}" >&2
      return 1
    fi
  done
}

validate_task_dirs() {
  local train_root="$1"
  local val_root="$2"
  local test_root="$3"

  require_split_dirs "${train_root}" train || return 1
  require_split_dirs "${val_root}" valid || return 1
  require_split_dirs "${test_root}" test || return 1
}

run_train() {
  local name="$1"
  local train_root="$2"
  local val_root="$3"
  local test_root="$4"
  local seed="$5"
  local batch_size="$6"
  shift 6

  local log_file="runs/logs/run_${name}.log"

  if ! validate_task_dirs "${train_root}" "${val_root}" "${test_root}"; then
    FAILED_TASKS+=("${name}:dataset")
    return
  fi

  echo "============================================================"
  echo "Starting ${name}"
  echo "Train root: ${train_root}/train"
  echo "Valid root: ${val_root}/valid"
  echo "Test root:  ${test_root}/test"
  echo "Seed: ${seed}"
  echo "Batch size: ${batch_size}"
  echo "Log: ${log_file}"
  echo "============================================================"

  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  if python train.py "${COMMON_ARGS[@]}" \
    --train-img-dir "${train_root}/train/images" \
    --train-mask-dir "${train_root}/train/masks" \
    --val-img-dir "${val_root}/valid/images" \
    --val-mask-dir "${val_root}/valid/masks" \
    --test-img-dir "${test_root}/test/images" \
    --test-mask-dir "${test_root}/test/masks" \
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

eval_b_runs() {
  local label="$1"
  local test_root="$2"
  local tta_mode="$3"

  if ! require_split_dirs "${test_root}" test; then
    echo "Skipping ${label}: test split not found under ${test_root}"
    return
  fi

  echo "============================================================"
  echo "Evaluating B4-series on ${label}"
  echo "Test root: ${test_root}"
  echo "TTA: ${tta_mode}"
  echo "============================================================"

  if python scripts/evaluate_history_on_test.py \
    --run-roots runs \
    --include-run-pattern "B4_*" \
    --test-img-dir "${test_root}/test/images" \
    --test-mask-dir "${test_root}/test/masks" \
    --output-csv "runs/debug_eval/${label}_B4_history_eval.csv" \
    --output-json "runs/debug_eval/${label}_B4_history_eval.json" \
    --device cuda \
    --batch-size 8 \
    --num-workers 4 \
    --threshold 0.5 \
    --tta "${tta_mode}" \
    --encoder-weights none \
    --default-encoder-weights none; then
    echo "Finished evaluation: ${label}"
  else
    local rc=$?
    echo "FAILED evaluation ${label} (exit ${rc}); continuing." >&2
    FAILED_TASKS+=("eval_${label}:exit_${rc}")
  fi
}

if [[ "${RUN_B}" == "1" ]]; then
  # B1: primary anchor. This reuses the historically strongest recipe, but
  # trains and validates only on B.
  run_train "B4_anchor_scale075_cls2w125_os15_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5

  # B2: loss-weight ablation. If B annotations are cleaner, class2 may not need
  # extra loss pressure once oversampling is enabled.
  run_train "B4_scale075_cls1111_os15_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5

  # B3: class2 stronger-weight ablation. Keep only if global Dice and class2
  # both improve on curated B.
  run_train "B4_scale075_cls2w15_os15_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.5 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5

  # B4: oversampling strength ablation. Tests whether clean B benefits from
  # more class2 exposure without changing loss weights.
  run_train "B4_scale075_cls2w125_os20_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 2.0

  # B5: resolution ablation. Use smaller batch to fit memory; keep only if the
  # curated B global Dice gain justifies slower training/inference.
  run_train "B4_fullres_cls2w125_os15_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5
else
  echo "Skipping B pilots (RUN_B=${RUN_B})."
fi

if [[ "${RUN_EXTRA}" == "1" ]]; then
  # Extra runs: enable after the core B1-B5 results are inspected.
  run_train "B4_scale075_cls2w125_noos_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1

  run_train "B4_scale075_cls2w125_os15_notta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5 \
    --val-tta none \
    --test-tta none

  run_train "B4_scale075_cls2w125_os15_wd1e3_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5 \
    --weight-decay 1e-3

  run_train "B4_scale075_cls2w125_os15_dropout03_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5 \
    --dropout-rate 0.3

  run_train "B4_scale075_cls2w125_os15_resnet50_tta_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5 \
    --encoder-name resnet50

  run_train "B4_scale075_cls2w125_os15_tta_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 \
    --oversample-factor 1.5
else
  echo "Skipping extra pilots (set RUN_EXTRA=1 to enable)."
fi

eval_b_runs "B_curated_test_tta" "${DATA_B_CURATED}" "flips"
eval_b_runs "B_curated_test_notta" "${DATA_B_CURATED}" "none"

if [[ "${RUN_REFERENCE_EVAL}" == "1" ]]; then
  eval_b_runs "B_original_test_tta_reference" "${DATA_B}" "flips"
else
  echo "Skipping raw B reference evaluation (RUN_REFERENCE_EVAL=${RUN_REFERENCE_EVAL})."
fi

echo "============================================================"
if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "Completed with ${#FAILED_TASKS[@]} failed task(s):"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  echo "Inspect runs/logs/run_B4_*.log and runs/debug_eval/*_B4_history_eval.csv."
  exit 1
fi

echo "All requested B4-series tasks completed successfully."
echo "Primary metric: 4-class global Dice on B_curated_test_tta."
echo "Training logs: runs/logs/run_B4_*.log"
echo "Run directories: runs/B4_*"
echo "Evaluation outputs: runs/debug_eval/*_B4_history_eval.csv"
