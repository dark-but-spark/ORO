#!/bin/bash
set -uo pipefail

# Run on the server from MultiResUNet:
#   cd ~/ORO/MultiResUNet
#   bash ../temp.sh
#
# Default plan (4 runs):
#   Two seeds x {original train/valid, manually filtered train/valid}.
#   All runs use the same original A test and exactly the same model settings.
#
# Optional model pilots (2 extra runs, together with the 4 controlled runs):
#   RUN_MODEL_PILOTS=1 bash ../temp.sh
# Run only the optional pilots after the controlled runs already finished:
#   RUN_CONTROLLED=0 RUN_MODEL_PILOTS=1 bash ../temp.sh
#
# The script continues after a failed run and prints a failure summary at the end.

mkdir -p runs/logs runs/debug_eval

DATA_A="../data/20260204111923"
DATA_A_FILTERED="../data/20260204111923_trainval_review_filtered_20260731"
DATA_A_CURATED="../data/20260204111923_curated_manual_review_20260731"
DATA_B_CLEAN="../data/385-liver.groupclean.v1"
RUN_CONTROLLED="${RUN_CONTROLLED:-1}"
RUN_MODEL_PILOTS="${RUN_MODEL_PILOTS:-0}"

FAILED_TASKS=()

COMMON_ARGS=(
  --split-mode fixed
  --input-channels 3
  --output-channels 4
  --epochs 160
  --batch-size 16
  --scale
  --scale-factor 0.75
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
  --encoder-name resnet34
  --encoder-weights imagenet
  --dropout-rate 0.2
  --lr-scheduler cosine
  --early-stopping-min-epochs 80
  --early-stopping-patience 30
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

validate_dataset() {
  local root="$1"
  local split
  local kind

  for split in train valid; do
    for kind in images masks; do
      if [[ ! -d "${root}/${split}/${kind}" ]]; then
        echo "Missing dataset directory: ${root}/${split}/${kind}" >&2
        return 1
      fi
    done
  done

  if [[ ! -d "${DATA_A}/test/images" || ! -d "${DATA_A}/test/masks" ]]; then
    echo "Missing fixed original A test under ${DATA_A}/test" >&2
    return 1
  fi
}

run_train() {
  local name="$1"
  local data_root="$2"
  local seed="$3"
  shift 3

  local log_file="runs/logs/run_${name}.log"

  if ! validate_dataset "${data_root}"; then
    FAILED_TASKS+=("${name}:dataset")
    return
  fi

  echo "============================================================"
  echo "Starting ${name}"
  echo "Train/valid root: ${data_root}"
  echo "Fixed test root: ${DATA_A}"
  echo "Seed: ${seed}"
  echo "Log: ${log_file}"
  echo "============================================================"

  if python train.py "${COMMON_ARGS[@]}" \
    --train-img-dir "${data_root}/train/images" \
    --train-mask-dir "${data_root}/train/masks" \
    --val-img-dir "${data_root}/valid/images" \
    --val-mask-dir "${data_root}/valid/masks" \
    --test-img-dir "${DATA_A}/test/images" \
    --test-mask-dir "${DATA_A}/test/masks" \
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

eval_y_runs() {
  local label="$1"
  local test_root="$2"

  if [[ ! -d "${test_root}/test/images" || ! -d "${test_root}/test/masks" ]]; then
    echo "Skipping ${label}: test directory not found: ${test_root}"
    return
  fi

  echo "============================================================"
  echo "Evaluating Y-series on ${label}"
  echo "Test root: ${test_root}"
  echo "============================================================"

  if python scripts/evaluate_history_on_test.py \
    --run-roots runs \
    --include-run-pattern "Y_*" \
    --test-img-dir "${test_root}/test/images" \
    --test-mask-dir "${test_root}/test/masks" \
    --output-csv "runs/debug_eval/${label}_Y_history_eval.csv" \
    --output-json "runs/debug_eval/${label}_Y_history_eval.json" \
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

# Phase 1: controlled data-cleaning comparison.
# Do not infer a cleaning gain from one seed; both pairs must be compared.
if [[ "${RUN_CONTROLLED}" == "1" ]]; then
  run_train "Y_A_original_scale075_seed42" "${DATA_A}" 42
  run_train "Y_A_filtered_scale075_seed42" "${DATA_A_FILTERED}" 42
  run_train "Y_A_original_scale075_seed44" "${DATA_A}" 44
  run_train "Y_A_filtered_scale075_seed44" "${DATA_A_FILTERED}" 44
else
  echo "Skipping controlled data comparison (RUN_CONTROLLED=${RUN_CONTROLLED})."
fi

# Phase 2: optional single-variable model pilots on the filtered dataset.
# Enable explicitly. To avoid repeating completed controlled runs:
#   RUN_CONTROLLED=0 RUN_MODEL_PILOTS=1 bash ../temp.sh
if [[ "${RUN_MODEL_PILOTS}" == "1" ]]; then
  run_train "Y_A_filtered_scale075_dropout03" "${DATA_A_FILTERED}" 42 \
    --dropout-rate 0.3

  run_train "Y_A_filtered_scale075_plateau" "${DATA_A_FILTERED}" 42 \
    --lr-scheduler plateau \
    --lr-patience 8
else
  echo "Skipping optional model pilots (set RUN_MODEL_PILOTS=1 to enable)."
fi

# Final evaluation: primary original test, explanatory curated test, and
# secondary cross-source B_clean test. B_clean must not drive A hyperparameters.
eval_y_runs "A_original_test" "${DATA_A}"
eval_y_runs "A_curated_test" "${DATA_A_CURATED}"
eval_y_runs "Bclean_cross_source_test" "${DATA_B_CLEAN}"

echo "============================================================"
if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "Completed with ${#FAILED_TASKS[@]} failed task(s):"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  echo "Inspect runs/logs/run_Y_*.log and the timestamped runs/*/logs directory."
  exit 1
fi

echo "All requested Y-series tasks completed successfully."
echo "Training logs: runs/logs/run_Y_*.log"
echo "Run directories: runs/Y_*"
echo "Evaluation outputs: runs/debug_eval/*_Y_history_eval.csv"
