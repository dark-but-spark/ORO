#!/bin/bash
set -uo pipefail

# Run on the server from MultiResUNet:
#   cd ~/ORO/MultiResUNet
#   bash ../temp.sh
#
# Purpose:
#   Z-series controlled pilots after manual review.
#   - A line: train on original A train, compare original valid vs high-quality valid.
#   - B line: B-only high-score pilots, trained sequentially, not in parallel.
#   - Evaluation: every Z checkpoint is re-scored on A original test,
#     A high-quality test, B original test, and B curated test.
#
# Optional expansion after default runs finish:
#   RUN_EXTRA=1 bash ../temp.sh

mkdir -p runs/logs runs/debug_eval

DATA_A="../data/20260204111923"
DATA_A_HQ="../data/20260204111923_high_quality_eval_20260804"
DATA_B="../data/385-liver.groupclean.v1"
DATA_B_CURATED="../data/385-liver.groupclean.v1_curated_eval_20260802"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_EXTRA="${RUN_EXTRA:-0}"

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
  --model-architecture smp_unet
  --encoder-name resnet34
  --encoder-weights imagenet
  --dropout-rate 0.2
  --lr-scheduler cosine
  --early-stopping-min-epochs 90
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

eval_z_runs() {
  local label="$1"
  local test_root="$2"

  if ! require_split_dirs "${test_root}" test; then
    echo "Skipping ${label}: test split not found under ${test_root}"
    return
  fi

  echo "============================================================"
  echo "Evaluating Z-series on ${label}"
  echo "Test root: ${test_root}"
  echo "============================================================"

  if python scripts/evaluate_history_on_test.py \
    --run-roots runs \
    --include-run-pattern "Z_*" \
    --test-img-dir "${test_root}/test/images" \
    --test-mask-dir "${test_root}/test/masks" \
    --output-csv "runs/debug_eval/${label}_Z_history_eval.csv" \
    --output-json "runs/debug_eval/${label}_Z_history_eval.json" \
    --device cuda \
    --batch-size 8 \
    --num-workers 4 \
    --threshold 0.5 \
    --tta none \
    --metric-ignore-classes 2 \
    --encoder-weights none \
    --default-encoder-weights none; then
    echo "Finished evaluation: ${label}"
  else
    local rc=$?
    echo "FAILED evaluation ${label} (exit ${rc}); continuing." >&2
    FAILED_TASKS+=("eval_${label}:exit_${rc}")
  fi
}

if [[ "${RUN_A}" == "1" ]]; then
  # A1: original validation anchor. This preserves comparability with older A runs.
  run_train "Z_A_origval_scale075_cls1111_seed42" "${DATA_A}" "${DATA_A}" "${DATA_A}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1 1

  # A2: clean validation signal. Train set is unchanged; only validation/test
  # switch to manually reviewed high-quality A subsets.
  run_train "Z_A_hqval_scale075_cls1111_seed42" "${DATA_A}" "${DATA_A_HQ}" "${DATA_A_HQ}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1 1

  # A3: reduce inflammation/class2 pressure because manual review found many
  # ambiguous or incomplete class2 annotations.
  run_train "Z_A_hqval_scale075_cls2w05_seed42" "${DATA_A}" "${DATA_A_HQ}" "${DATA_A_HQ}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 0.5 1
else
  echo "Skipping A pilots (RUN_A=${RUN_A})."
fi

if [[ "${RUN_B}" == "1" ]]; then
  # B runs are intentionally sequential and few. B_clean is small, so do not
  # add many knobs before the fullres/scale choice is clear.
  run_train "Z_B_curated_fullres_cls1111_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --class-weights 1 1 1 1

  run_train "Z_B_curated_scale075_cls1111_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1 1
else
  echo "Skipping B pilots (RUN_B=${RUN_B})."
fi

if [[ "${RUN_EXTRA}" == "1" ]]; then
  # Extra runs: enable only after the default five runs are inspected.
  run_train "Z_A_hqval_scale075_cls2w025_seed42" "${DATA_A}" "${DATA_A_HQ}" "${DATA_A_HQ}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 0.25 1

  run_train "Z_A_hqval_scale075_plateau_seed42" "${DATA_A}" "${DATA_A_HQ}" "${DATA_A_HQ}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 0.5 1 \
    --lr-scheduler plateau \
    --lr-patience 8

  run_train "Z_B_curated_fullres_dropout03_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --class-weights 1 1 1 1 \
    --dropout-rate 0.3

  run_train "Z_B_curated_fullres_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 8 \
    --class-weights 1 1 1 1
else
  echo "Skipping extra pilots (set RUN_EXTRA=1 to enable)."
fi

eval_z_runs "A_original_test" "${DATA_A}"
eval_z_runs "A_high_quality_test" "${DATA_A_HQ}"
eval_z_runs "B_original_test" "${DATA_B}"
eval_z_runs "B_curated_test" "${DATA_B_CURATED}"

echo "============================================================"
if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "Completed with ${#FAILED_TASKS[@]} failed task(s):"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  echo "Inspect runs/logs/run_Z_*.log and runs/debug_eval/*_Z_history_eval.csv."
  exit 1
fi

echo "All requested Z-series tasks completed successfully."
echo "Training logs: runs/logs/run_Z_*.log"
echo "Run directories: runs/Z_*"
echo "Evaluation outputs: runs/debug_eval/*_Z_history_eval.csv"
