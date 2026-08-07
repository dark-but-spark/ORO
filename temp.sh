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
#   - Primary validation/test: class-complete, group-disjoint B curated split.
#   - Original B evaluation is reference only because the raw split had leakage.
#
# Default: re-evaluate existing B4-B8 checkpoints on the balanced v2 split.
# No training queue runs unless its RUN_* switch is explicitly enabled.
#   bash ../temp.sh
#
# Optional historical pilot groups:
#   RUN_B=1 bash ../temp.sh
#   RUN_EXTRA=1 bash ../temp.sh
#   RUN_NIGHT=1 RUN_B6=0 bash ../temp.sh
#   RUN_B9=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B9_LOCKED_TEST=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B10=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B11=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B12=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B12_FINETUNE=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B11_ENSEMBLE=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B12_OVERNIGHT=1 bash ../temp.sh
#   RUN_B14_HALFDAY=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#   RUN_B15=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
#
# Optional reference-only evaluation on raw B:
#   RUN_REFERENCE_EVAL=1 bash ../temp.sh

mkdir -p runs/logs runs/debug_eval

DATA_B="../data/385-liver.groupclean.v1"
DATA_B_CURATED="../data/385-liver.groupclean.v1_curated_eval_balanced_20260806"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_B="${RUN_B:-0}"
RUN_EXTRA="${RUN_EXTRA:-0}"
RUN_NIGHT="${RUN_NIGHT:-0}"
RUN_B6="${RUN_B6:-0}"
RUN_B7="${RUN_B7:-0}"
RUN_ROI_PATCH="${RUN_ROI_PATCH:-0}"
RUN_B9="${RUN_B9:-0}"
RUN_B9_LOCKED_TEST="${RUN_B9_LOCKED_TEST:-0}"
RUN_B10="${RUN_B10:-0}"
RUN_B11="${RUN_B11:-0}"
RUN_B12="${RUN_B12:-0}"
RUN_B12_FINETUNE="${RUN_B12_FINETUNE:-0}"
RUN_B13="${RUN_B13:-0}"
RUN_B11_ENSEMBLE="${RUN_B11_ENSEMBLE:-0}"
RUN_B12_OVERNIGHT="${RUN_B12_OVERNIGHT:-0}"
RUN_B14_HALFDAY="${RUN_B14_HALFDAY:-0}"
RUN_B15="${RUN_B15:-0}"
RUN_CORRECTED_HISTORY_EVAL="${RUN_CORRECTED_HISTORY_EVAL:-0}"
RUN_BALANCED_HISTORY_EVAL="${RUN_BALANCED_HISTORY_EVAL:-1}"
RUN_REFERENCE_EVAL="${RUN_REFERENCE_EVAL:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

# One-switch extended overnight queue: 20 training jobs plus broad valid-only
# threshold/TTA evaluation and the existing B11 top-3 ensemble baseline.
# Explicit RUN_* values remain available for shorter follow-up queues.
if [[ "${RUN_B12_OVERNIGHT}" == "1" ]]; then
  RUN_B12=1
  RUN_B12_FINETUNE=1
  RUN_B13=1
  RUN_B11_ENSEMBLE=1
  RUN_BALANCED_HISTORY_EVAL=0
fi

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
  --eval-batch-size "${EVAL_BATCH_SIZE}"
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
  local run_pattern="$4"
  local threshold="${5:-0.5}"
  local split="${6:-test}"

  if ! require_split_dirs "${test_root}" "${split}"; then
    echo "Skipping ${label}: ${split} split not found under ${test_root}"
    return
  fi

  echo "============================================================"
  echo "Evaluating ${run_pattern} on ${label}"
  echo "Evaluation root: ${test_root}/${split}"
  echo "TTA: ${tta_mode}"
  echo "============================================================"

  if python scripts/evaluate_history_on_test.py \
    --run-roots runs \
    --include-run-pattern "${run_pattern}" \
    --test-img-dir "${test_root}/${split}/images" \
    --test-mask-dir "${test_root}/${split}/masks" \
    --output-csv "runs/debug_eval/${label}_history_eval.csv" \
    --output-json "runs/debug_eval/${label}_history_eval.json" \
    --device cuda \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers 4 \
    --threshold "${threshold}" \
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

eval_wide_valid_sweep() {
  local family="$1"
  local run_pattern="$2"
  local threshold_spec
  local label
  local threshold

  # Broad calibration range. Keep this on valid only; never use it to search
  # thresholds on the locked test split.
  for threshold_spec in \
    "0p35:0.35" "0p40:0.40" "0p45:0.45" "0p50:0.50" \
    "0p55:0.55" "0p60:0.60" "0p65:0.65"; do
    label="${threshold_spec%%:*}"
    threshold="${threshold_spec##*:}"
    eval_b_runs "B_balanced_v2_valid_tta_${family}_thr${label}" \
      "${DATA_B_CURATED}" "flips" "${run_pattern}" "${threshold}" "valid"
  done

  # Quantify whether flip TTA is actually helping instead of assuming it does.
  eval_b_runs "B_balanced_v2_valid_notta_${family}_thr0p45" \
    "${DATA_B_CURATED}" "none" "${run_pattern}" "0.45" "valid"
  eval_b_runs "B_balanced_v2_valid_notta_${family}_thr0p50" \
    "${DATA_B_CURATED}" "none" "${run_pattern}" "0.50" "valid"
}

eval_fine_valid_sweep() {
  local family="$1"
  local run_pattern="$2"
  local threshold_spec
  local label
  local threshold

  # Fine calibration around the useful range found by B12/B13.
  for threshold_spec in \
    "0p400:0.400" "0p425:0.425" "0p450:0.450" \
    "0p475:0.475" "0p500:0.500" "0p525:0.525"; do
    label="${threshold_spec%%:*}"
    threshold="${threshold_spec##*:}"
    eval_b_runs "B_balanced_v2_valid_tta_${family}_thr${label}" \
      "${DATA_B_CURATED}" "flips" "${run_pattern}" "${threshold}" "valid"
  done

  eval_b_runs "B_balanced_v2_valid_notta_${family}_thr0p450" \
    "${DATA_B_CURATED}" "none" "${run_pattern}" "0.450" "valid"
  eval_b_runs "B_balanced_v2_valid_notta_${family}_thr0p500" \
    "${DATA_B_CURATED}" "none" "${run_pattern}" "0.500" "valid"
}

latest_run_dir() {
  local pattern="$1"
  local matches=()
  shopt -s nullglob
  matches=(runs/${pattern})
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${matches[$((${#matches[@]} - 1))]}"
}

eval_b_ensemble() {
  local label="$1"
  local threshold="$2"
  shift 2
  local run_args=()
  local pattern
  local run_dir

  for pattern in "$@"; do
    if ! run_dir="$(latest_run_dir "${pattern}")"; then
      echo "Missing ensemble member matching runs/${pattern}" >&2
      FAILED_TASKS+=("ensemble_${label}:missing_${pattern}")
      return
    fi
    run_args+=(--run-dir "${run_dir}")
  done

  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  if python scripts/evaluate_ensemble_on_split.py \
    "${run_args[@]}" \
    --img-dir "${DATA_B_CURATED}/valid/images" \
    --mask-dir "${DATA_B_CURATED}/valid/masks" \
    --output-json "runs/debug_eval/${label}.json" \
    --device cuda \
    --batch-size 2 \
    --num-workers 4 \
    --evaluation-scale 0.75 \
    --thresholds "${threshold}" \
    --tta flips; then
    echo "Finished ensemble evaluation: ${label}"
  else
    local rc=$?
    echo "FAILED ensemble ${label} (exit ${rc}); continuing." >&2
    FAILED_TASKS+=("ensemble_${label}:exit_${rc}")
  fi
}

if [[ "${RUN_BALANCED_HISTORY_EVAL}" == "1" ]]; then
  # Establish a class-complete comparison table before any further search.
  # Keep threshold fixed at 0.5; never tune thresholds on the test split.
  eval_b_runs "B_balanced_v2_test_tta_B4_B8" "${DATA_B_CURATED}" "flips" "B[45678]_*"
  eval_b_runs "B_balanced_v2_test_notta_B4_B8" "${DATA_B_CURATED}" "none" "B[45678]_*"
  eval_b_runs "B_balanced_v2_valid_tta_B4_B8" "${DATA_B_CURATED}" "flips" "B[45678]_*" "0.5" "valid"
else
  echo "Skipping balanced v2 B4-B8 history evaluation."
fi

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

if [[ "${RUN_NIGHT}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B5 overnight queue.  Primary target is B curated-test 4-class GLOBAL Dice.
  # The reference to beat is B4_scale075_cls2w125_os20 (TTA Dice 0.825411).
  # Every job stays B-only and uses the same curated validation/test split.
  # -------------------------------------------------------------------------

  # Direction 1: loss balance.  B4 used BCE:Dice = 0.7:0.3.  Increase Dice
  # pressure to improve shape/overlap without changing the sample distribution.
  run_train "B5_scale075_os20_bce06_dice04_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --bce-weight 0.6 --dice-weight 0.4

  run_train "B5_scale075_os20_bce05_dice05_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --bce-weight 0.5 --dice-weight 0.5

  # Direction 2: identify the useful class-2 sampling range around 2.0.
  run_train "B5_scale075_cls2w125_os175_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 1.75

  run_train "B5_scale075_cls2w125_os225_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.25

  run_train "B5_scale075_cls2w125_os250_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.5

  # Direction 3: keep sampling at its current optimum, then tune only the
  # class-2 loss weight.  This distinguishes loss bias from data exposure.
  run_train "B5_scale075_cls2w115_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B5_scale075_cls2w135_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.35 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  # Direction 4: Focal + Dice is a qualitatively different loss.  It can help
  # hard/small regions but is retained only if it improves GLOBAL Dice.
  run_train "B5_scale075_focal05_g15_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --use-focal-loss --focal-alpha 0.5 --focal-gamma 1.5 \
    --bce-weight 0.6 --dice-weight 0.4

  # Direction 5: capacity/regularization sentinel.  One ResNet-50 run tests
  # whether the B-only data supports a larger encoder; do not expand it unless
  # it clears the ResNet-34 reference.
  run_train "B5_scale075_resnet50_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50

  # Direction 6: reproducibility.  Repeat the current winner with two unseen
  # seeds.  A tiny one-seed gain is not a promotion candidate.
  run_train "B5_scale075_cls2w125_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B5_scale075_cls2w125_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.25 1 \
    --oversample-class-indices 2 --oversample-factor 2.0
else
  echo "Skipping overnight B5 queue (RUN_NIGHT=${RUN_NIGHT})."
fi

if [[ "${RUN_NIGHT}" == "1" ]]; then
  eval_b_runs "B_curated_test_tta_B5" "${DATA_B_CURATED}" "flips" "B5_*"
  eval_b_runs "B_curated_test_notta_B5" "${DATA_B_CURATED}" "none" "B5_*"
fi

if [[ "${RUN_B6}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B6 overnight queue. Primary target is B curated-test 4-class GLOBAL Dice.
  # B5 winner: cls2w=1.15, os=2.0, seed=42, TTA Dice=0.832412.
  # B5 showed that larger Dice-loss weight, focal loss, and os > 2.0 were not
  # consistently useful. B6 therefore verifies the winner first, then searches
  # only its local neighborhood plus a few independent training directions.
  # -------------------------------------------------------------------------

  # Priority 1: reproducibility of the new winner. These two runs decide
  # whether 0.8324 is a stable recipe or a favorable seed.
  run_train "B6_anchor_cls2w115_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B6_anchor_cls2w115_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  # Direction 1: fine class-2 loss-weight search around 1.15.
  run_train "B6_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B6_cls2w110_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.10 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B6_cls2w120_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.20 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  # Direction 2: local sampling search with the new 1.15 class weight.
  run_train "B6_cls2w115_os18_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 1.8

  run_train "B6_cls2w115_os22_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.2

  # Direction 3: optimization speed. Keep all data/loss settings fixed.
  run_train "B6_cls2w115_os20_lr1e5_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --learning-rate 1e-5

  run_train "B6_cls2w115_os20_lr3e5_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --learning-rate 3e-5

  # Direction 4: augmentation regularization. B images may need less or more
  # invariance than the current cosine curriculum maximum of 0.4.
  run_train "B6_cls2w115_os20_aug02_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --curriculum-max-aug-level 0.2

  run_train "B6_cls2w115_os20_aug06_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --curriculum-max-aug-level 0.6

  # Direction 5: capacity. ResNet50 was second-best in B5 at 0.82664 despite
  # using the older class weight, so combine it with the new 1.15 setting.
  run_train "B6_resnet50_cls2w115_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50
else
  echo "Skipping B6 queue (RUN_B6=${RUN_B6})."
fi

if [[ "${RUN_B6}" == "1" ]]; then
  eval_b_runs "B_curated_test_tta_B6" "${DATA_B_CURATED}" "flips" "B6_*"
  eval_b_runs "B_curated_test_notta_B6" "${DATA_B_CURATED}" "none" "B6_*"
fi

if [[ "${RUN_CORRECTED_HISTORY_EVAL}" == "1" ]]; then
  # Previous history evaluation averaged batches equally. With 43 test images
  # and batch size 8, the final 3-image batch was overweighted. The evaluator
  # now weights every image equally, so regenerate the comparable B4-B6 table
  # before promoting any recipe.
  eval_b_runs "B_curated_test_tta_B4_B6_sampleweighted" "${DATA_B_CURATED}" "flips" "B[456]_*"
  eval_b_runs "B_curated_test_notta_B4_B6_sampleweighted" "${DATA_B_CURATED}" "none" "B[456]_*"
else
  echo "Skipping corrected B4-B6 history evaluation."
fi

if [[ "${RUN_B7}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B7 overnight queue. Primary target: sample-weighted, per-image 4-class
  # global Dice on curated B. B6's apparent peak (cls2w=1.05, seed=42,
  # old TTA Dice=0.834425) is not promoted until it reproduces across seeds.
  # -------------------------------------------------------------------------

  # Priority 1: reproduce the B6 1.05 winner. Together with existing seed 42,
  # these runs provide a four-seed stability estimate.
  run_train "B7_cls2w105_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B7_cls2w105_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B7_cls2w105_os20_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  # Priority 2: determine whether class-2 weighting should return toward 1.0.
  run_train "B7_cls2w100_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.00 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B7_cls2w1025_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.025 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  run_train "B7_cls2w1075_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.075 1 \
    --oversample-class-indices 2 --oversample-factor 2.0

  # Priority 3: check the sampling interaction only around the new 1.05 center.
  run_train "B7_cls2w105_os18_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 1.8

  run_train "B7_cls2w105_os22_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.2

  # Priority 4: B6 showed 3e-5 is viable and converges faster. Test whether it
  # combines with the 1.05 weight before spending more seeds on it.
  run_train "B7_cls2w105_os20_lr3e5_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --learning-rate 3e-5

  # Priority 5: capacity verification. ResNet50 led no-TTA in B6, so test the
  # new class weight once and reproduce the existing 1.15 recipe twice.
  run_train "B7_resnet50_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50

  run_train "B7_resnet50_cls2w115_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50

  run_train "B7_resnet50_cls2w115_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.15 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50
else
  echo "Skipping B7 queue (RUN_B7=${RUN_B7})."
fi

if [[ "${RUN_B7}" == "1" ]]; then
  eval_b_runs "B_curated_test_tta_B7_sampleweighted" "${DATA_B_CURATED}" "flips" "B7_*"
  eval_b_runs "B_curated_test_notta_B7_sampleweighted" "${DATA_B_CURATED}" "none" "B7_*"

  # Threshold selection must use validation rather than the test set. These
  # exports are intentionally labeled VALID; choose one threshold after B7,
  # then run exactly one locked test evaluation in the next stage.
  for threshold in 0.40 0.45 0.50 0.55 0.60; do
    threshold_label="${threshold/./p}"
    eval_b_runs "B_curated_valid_tta_B7_thr${threshold_label}" "${DATA_B_CURATED}" "flips" "B7_*" "${threshold}" "valid"
  done
fi

if [[ "${RUN_ROI_PATCH}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B8 native-resolution ROI/patch queue.
  # Train: mask-guided patches cropped from the original 640x640 B images.
  # Valid/test: complete 640x640 images; no GT-guided cropping is used outside
  # training. Validation TTA is disabled to keep full-resolution training fast;
  # final history evaluation below reports both flips TTA and no-TTA.
  # -------------------------------------------------------------------------

  # Main ROI recipe and one unseen seed for immediate stability evidence.
  run_train "B8_roi384_pos075_all_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --train-patch-size 384 \
    --patch-positive-probability 0.75 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  run_train "B8_roi384_pos075_all_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --train-patch-size 384 \
    --patch-positive-probability 0.75 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  # Patch-size search: 320 emphasizes detail; 448 retains more context.
  run_train "B8_roi320_pos075_all_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 20 \
    --train-patch-size 320 \
    --patch-positive-probability 0.75 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  run_train "B8_roi448_pos075_all_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --train-patch-size 448 \
    --patch-positive-probability 0.75 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  # Foreground/background balance around the 384 mainline.
  run_train "B8_roi384_pos050_all_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --train-patch-size 384 \
    --patch-positive-probability 0.50 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  run_train "B8_roi384_pos090_all_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --train-patch-size 384 \
    --patch-positive-probability 0.90 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  # Class-2 targeted ROI, because class 2 occupies only about 0.85% of B train
  # pixels. Images without class 2 automatically fall back to random patches.
  run_train "B8_roi384_pos085_cls2_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --train-patch-size 384 \
    --patch-positive-probability 0.85 \
    --patch-class-indices 2 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none

  # Random-patch control determines how much gain comes from ROI guidance rather
  # than native resolution alone.
  run_train "B8_patch384_random_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --train-patch-size 384 \
    --patch-positive-probability 0.0 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none
else
  echo "Skipping ROI/patch queue (RUN_ROI_PATCH=${RUN_ROI_PATCH})."
fi

if [[ "${RUN_ROI_PATCH}" == "1" ]]; then
  eval_b_runs "B_curated_test_tta_B8_roi_sampleweighted" "${DATA_B_CURATED}" "flips" "B8_*"
  eval_b_runs "B_curated_test_notta_B8_roi_sampleweighted" "${DATA_B_CURATED}" "none" "B8_*"

  # Threshold selection stays on validation. Keep the first sweep coarse; once
  # the winning patch recipe is known, refine only around its best threshold.
  for threshold in 0.45 0.50 0.55; do
    threshold_label="${threshold/./p}"
    eval_b_runs "B_curated_valid_tta_B8_roi_thr${threshold_label}" "${DATA_B_CURATED}" "flips" "B8_*" "${threshold}" "valid"
  done
fi

if [[ "${RUN_B9}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B9 balanced-v2 queue.
  # Current balanced-v2 TTA leader: B6_cls2w105_os20_seed42, Dice=0.838671.
  # Main weakness remains class 2 (about 0.77-0.79 Dice in the best runs).
  # These jobs do not run test automatically; rank them on balanced-v2 valid,
  # then run one locked test evaluation for the shortlisted recipe.
  # -------------------------------------------------------------------------

  # Direction 1: reproduce the balanced-v2 leader across new seeds.
  run_train "B9_cls2w105_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B9_cls2w105_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B9_cls2w105_os20_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # Direction 2: class-2 local pressure.  The 1.10 run had the best class-2
  # Dice among the top balanced-v2 models, while global Dice stayed close.
  run_train "B9_cls2w110_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.10 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B9_cls2w110_os20_dice04_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.10 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --bce-weight 0.6 --dice-weight 0.4 \
    --no-test-after-training

  # Direction 3: slightly more class-2 exposure, but avoid the older heavy
  # oversampling range that did not clearly improve global Dice.
  run_train "B9_cls2w105_os24_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.4 \
    --no-test-after-training

  # Direction 4: ROI/context follow-up.  ROI448 is close to the whole-image
  # leader, so verify one new seed and one class-2-targeted variant.
  run_train "B9_roi448_pos075_all_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 12 \
    --train-patch-size 448 \
    --patch-positive-probability 0.75 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none \
    --no-test-after-training

  run_train "B9_roi448_pos085_cls2_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --train-patch-size 448 \
    --patch-positive-probability 0.85 \
    --patch-class-indices 2 \
    --patch-min-positive-pixels 32 \
    --patch-center-jitter 0.20 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --val-tta none \
    --no-test-after-training

  # Direction 5: modest capacity check.  ResNet50 is worth one more run with
  # the balanced-v2 leader's lower class-2 weight, but this is not the main bet.
  run_train "B9_resnet50_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50 \
    --no-test-after-training

  eval_b_runs "B_balanced_v2_valid_tta_B9_thr0p45" "${DATA_B_CURATED}" "flips" "B9_*" "0.45" "valid"
  eval_b_runs "B_balanced_v2_valid_tta_B9_thr0p50" "${DATA_B_CURATED}" "flips" "B9_*" "0.50" "valid"
  eval_b_runs "B_balanced_v2_valid_tta_B9_thr0p55" "${DATA_B_CURATED}" "flips" "B9_*" "0.55" "valid"
else
  echo "Skipping B9 balanced-v2 queue (RUN_B9=${RUN_B9})."
fi

if [[ "${RUN_B9_LOCKED_TEST}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # One-time locked balanced-v2 test for candidates selected on valid only.
  # Do not change these thresholds after looking at the test results, and do
  # not reuse this switch for repeated test-guided tuning.
  # -------------------------------------------------------------------------
  eval_b_runs "B_balanced_v2_test_tta_B9_cls2w105_seed44_thr0p50_locked" \
    "${DATA_B_CURATED}" "flips" "B9_cls2w105_os20_seed44_*" "0.50" "test"

  eval_b_runs "B_balanced_v2_test_tta_B9_cls2w110_seed43_thr0p55_locked" \
    "${DATA_B_CURATED}" "flips" "B9_cls2w110_os20_seed43_*" "0.55" "test"

  eval_b_runs "B_balanced_v2_test_tta_B9_resnet50_seed42_thr0p45_locked" \
    "${DATA_B_CURATED}" "flips" "B9_resnet50_cls2w105_os20_seed42_*" "0.45" "test"
else
  echo "Skipping one-time B9 locked test (RUN_B9_LOCKED_TEST=${RUN_B9_LOCKED_TEST})."
fi

if [[ "${RUN_B10}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B10 strengthening queue after B-data cleaning.
  #
  # B9 showed that wider class-2 weight/oversampling sweeps have saturated:
  # os=2.4 and Dice weight=0.4 regressed, pure ROI hurt global context, and
  # ordinary Focal was already weak in B5.  B10 therefore keeps the stable
  # cls2w=1.05/os=2.0 anchor and tests only new, supported directions.
  #
  # All jobs stop at validation.  Rank on balanced-v2 valid first; test only a
  # final pre-declared shortlist once.
  # -------------------------------------------------------------------------

  # Direction 1: retain more spatial detail than scale=0.75.  These two runs
  # test whether small/class-2 structures were being lost by down-scaling.
  run_train "B10_scale0875_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 12 \
    --scale --scale-factor 0.875 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B10_fullres_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # Direction 2: do not let tiny residual regions trigger whole-image
  # oversampling.  This emphasizes class-2 images with meaningful area while
  # preserving the same effective oversampling factor.
  run_train "B10_cls2w105_os20_minpix64_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --oversample-min-pixels 64 \
    --no-test-after-training

  run_train "B10_cls2w105_os20_minpix256_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --oversample-min-pixels 256 \
    --no-test-after-training

  # Direction 3: reduce augmentation pressure.  The cleaned B domain may
  # benefit from preserving its real appearance instead of adding variation.
  run_train "B10_cls2w105_os20_aug02_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 16 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --curriculum-max-aug-level 0.2 \
    --no-test-after-training

  # Direction 4: one stability check for the larger encoder.  The first
  # ResNet50 B9 run had the best class-2 valid Dice but not the best global
  # Dice; a second seed decides whether that signal is reproducible.
  run_train "B10_resnet50_cls2w105_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50 \
    --no-test-after-training

  # Coarse valid-only threshold sweep.  Promote a candidate only if global
  # Dice improves and no previously strong class collapses.
  eval_b_runs "B_balanced_v2_valid_tta_B10_thr0p45" "${DATA_B_CURATED}" "flips" "B10_*" "0.45" "valid"
  eval_b_runs "B_balanced_v2_valid_tta_B10_thr0p50" "${DATA_B_CURATED}" "flips" "B10_*" "0.50" "valid"
  eval_b_runs "B_balanced_v2_valid_tta_B10_thr0p55" "${DATA_B_CURATED}" "flips" "B10_*" "0.55" "valid"
else
  echo "Skipping B10 strengthening queue (RUN_B10=${RUN_B10})."
fi

if [[ "${RUN_B11}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B11 valid-only queue after reading B10.
  #
  # B10 says the strongest new signal is full resolution:
  # - B10_fullres improved class 2 clearly over the scale=0.75 seed42 anchor.
  # - scale=0.875 helped less, so the next search should verify fullres across
  #   seeds and only then decide whether to pay the compute cost.
  # - minpix64/minpix256 and weak augmentation regressed; do not continue them.
  # - ResNet50 is promising but not proven because its gain is seed/class0 heavy.
  #
  # All jobs stop at validation.  Promote by multi-seed mean, class-2 Dice, and
  # worst-class stability.  Keep test locked until a small shortlist is fixed.
  # -------------------------------------------------------------------------

  # Direction 1: full-resolution ResNet34 seed stability.  This is the main
  # B11 bet because B10_fullres raised class-2 Dice without code changes.
  run_train "B11_fullres_resnet34_cls2w105_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 8 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B11_fullres_resnet34_cls2w105_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 8 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # Direction 2: larger encoder at full resolution.  If memory is tight, lower
  # these two batch sizes from 6 to 4 before launching.
  run_train "B11_fullres_resnet50_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 6 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50 \
    --no-test-after-training

  run_train "B11_fullres_resnet50_cls2w105_os20_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 6 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50 \
    --no-test-after-training

  # Direction 3: regularization check for fullres.  B10_fullres had a larger
  # train-valid gap than the scale=0.75 anchor, so test weight decay before
  # increasing model size further.
  run_train "B11_fullres_resnet34_wd1e3_cls2w105_os20_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --weight-decay 1e-3 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # Direction 4: ResNet50 seed completion at the old scale.  This separates
  # encoder benefit from the full-resolution benefit.
  run_train "B11_scale075_resnet50_cls2w105_os20_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 12 \
    --scale --scale-factor 0.75 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --encoder-name resnet50 \
    --no-test-after-training

  # Coarse valid-only threshold sweep.  Do not use test here.
  eval_b_runs "B_balanced_v2_valid_tta_B11_thr0p45" "${DATA_B_CURATED}" "flips" "B11_*" "0.45" "valid"
  eval_b_runs "B_balanced_v2_valid_tta_B11_thr0p50" "${DATA_B_CURATED}" "flips" "B11_*" "0.50" "valid"
  eval_b_runs "B_balanced_v2_valid_tta_B11_thr0p55" "${DATA_B_CURATED}" "flips" "B11_*" "0.55" "valid"
else
  echo "Skipping B11 post-B10 queue (RUN_B11=${RUN_B11})."
fi

if [[ "${RUN_B12}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B12 high-upside valid-only queue.
  #
  # Pure ROI previously lost global context. The new mixed mode keeps whole
  # images in most samples and zooms a minority of class-2 ROI crops back to
  # the same network input size. This makes whole/ROI tensors batch-compatible.
  # Alternative decoders are isolated from ROI changes for clean attribution.
  # -------------------------------------------------------------------------

  run_train "B12_r50_s075_mixroi448_p020_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 10 \
    --scale --scale-factor 0.75 \
    --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --train-patch-size 448 \
    --patch-sampling-probability 0.20 --patch-resize-to-full \
    --patch-positive-probability 0.90 --patch-class-indices 2 \
    --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
    --no-test-after-training

  run_train "B12_r50_s075_mixroi448_p035_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 10 \
    --scale --scale-factor 0.75 \
    --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --train-patch-size 448 \
    --patch-sampling-probability 0.35 --patch-resize-to-full \
    --patch-positive-probability 0.90 --patch-class-indices 2 \
    --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
    --no-test-after-training

  run_train "B12_r50_s075_mixroi384_p025_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 10 \
    --scale --scale-factor 0.75 \
    --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --train-patch-size 384 \
    --patch-sampling-probability 0.25 --patch-resize-to-full \
    --patch-positive-probability 0.90 --patch-class-indices 2 \
    --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
    --no-test-after-training

  run_train "B12_unetpp_r50_s075_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 8 \
    --scale --scale-factor 0.75 \
    --model-architecture smp_unetplusplus --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B12_deeplabv3p_r50_s075_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 10 \
    --scale --scale-factor 0.75 \
    --model-architecture smp_deeplabv3plus --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B12_r50_s0875_seed43" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 43 8 \
    --scale --scale-factor 0.875 \
    --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  eval_wide_valid_sweep "B12" "B12_*"
else
  echo "Skipping B12 high-upside queue (RUN_B12=${RUN_B12})."
fi

if [[ "${RUN_B13}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B13 broad exploration matrix. These runs extend the search axes that B11
  # could not resolve: multi-seed stability, encoder/resolution interaction,
  # ROI dose/size, decoder family, learning rate, and class-2 exposure.
  # Every candidate is selected on curated valid only.
  # -------------------------------------------------------------------------

  # A. Complete the strongest scale=0.75 ResNet50 family across seeds.
  run_train "B13_r50_s075_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 12 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B13_r50_s075_seed46" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 46 12 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # B. Complete scale=0.875 ResNet50 across seeds; this tests whether the
  # larger encoder can use extra detail better than ResNet34 did in B10.
  run_train "B13_r50_s0875_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --scale --scale-factor 0.875 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B13_r50_s0875_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 8 \
    --scale --scale-factor 0.875 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # C. Extend mixed-ROI dose and crop-size coverage beyond B12.
  run_train "B13_r50_s075_mixroi448_p010_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 10 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --train-patch-size 448 --patch-sampling-probability 0.10 --patch-resize-to-full \
    --patch-positive-probability 0.90 --patch-class-indices 2 \
    --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
    --no-test-after-training

  run_train "B13_r50_s075_mixroi448_p030_seed46" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 46 10 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --train-patch-size 448 --patch-sampling-probability 0.30 --patch-resize-to-full \
    --patch-positive-probability 0.90 --patch-class-indices 2 \
    --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
    --no-test-after-training

  run_train "B13_r50_s075_mixroi512_p020_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 10 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --train-patch-size 512 --patch-sampling-probability 0.20 --patch-resize-to-full \
    --patch-positive-probability 0.90 --patch-class-indices 2 \
    --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
    --no-test-after-training

  # D. Decoder comparison with the smaller encoder separates decoder gains
  # from ResNet50 capacity gains already covered by B12.
  run_train "B13_unetpp_r34_s075_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 10 \
    --scale --scale-factor 0.75 \
    --model-architecture smp_unetplusplus --encoder-name resnet34 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B13_deeplabv3p_r34_s075_seed44" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 44 12 \
    --scale --scale-factor 0.75 \
    --model-architecture smp_deeplabv3plus --encoder-name resnet34 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # E. Optimization range around the current 2e-5 anchor.
  run_train "B13_r50_s075_lr1e5_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 12 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --learning-rate 1e-5 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  run_train "B13_r50_s075_lr3e5_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 12 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --learning-rate 3e-5 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 2.0 \
    --no-test-after-training

  # F. Fill the gap between no extra exposure and os=2.0; os=2.4 already
  # regressed in B9, so the useful unexplored direction is lower, not higher.
  run_train "B13_r50_s075_os175_seed45" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 45 12 \
    --scale --scale-factor 0.75 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 \
    --oversample-class-indices 2 --oversample-factor 1.75 \
    --no-test-after-training

  eval_wide_valid_sweep "B13" "B13_*"
else
  echo "Skipping B13 broad exploration queue (RUN_B13=${RUN_B13})."
fi

if [[ "${RUN_B12_FINETUNE}" == "1" ]]; then
  # Stage 2: preserve the best scale=0.75 ResNet50 solution and expose it to a
  # small proportion of zoomed class-2 ROI views at a 4x lower learning rate.
  if B11_R50_ANCHOR="$(latest_run_dir 'B11_scale075_resnet50_cls2w105_os20_seed43_*')"; then
    B11_R50_CHECKPOINT="${B11_R50_ANCHOR}/models/best_model.pth"

    run_train "B12ft_r50_s075_mixroi448_p020_seed46" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 46 10 \
      --scale --scale-factor 0.75 \
      --model-architecture smp_unet --encoder-name resnet50 --encoder-weights none \
      --init-checkpoint "${B11_R50_CHECKPOINT}" \
      --epochs 80 --learning-rate 5e-6 --lr-cosine-t-max 60 \
      --early-stopping-min-epochs 30 --early-stopping-patience 20 \
      --augmentation-curriculum none \
      --class-weights 1 1 1.05 1 \
      --oversample-class-indices 2 --oversample-factor 2.0 \
      --train-patch-size 448 \
      --patch-sampling-probability 0.20 --patch-resize-to-full \
      --patch-positive-probability 0.90 --patch-class-indices 2 \
      --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
      --no-test-after-training

    run_train "B12ft_r50_s075_mixroi384_p030_seed47" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 47 10 \
      --scale --scale-factor 0.75 \
      --model-architecture smp_unet --encoder-name resnet50 --encoder-weights none \
      --init-checkpoint "${B11_R50_CHECKPOINT}" \
      --epochs 80 --learning-rate 5e-6 --lr-cosine-t-max 60 \
      --early-stopping-min-epochs 30 --early-stopping-patience 20 \
      --augmentation-curriculum none \
      --class-weights 1 1 1.05 1 \
      --oversample-class-indices 2 --oversample-factor 2.0 \
      --train-patch-size 384 \
      --patch-sampling-probability 0.30 --patch-resize-to-full \
      --patch-positive-probability 0.90 --patch-class-indices 2 \
      --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
      --no-test-after-training

    eval_wide_valid_sweep "B12ft" "B12ft_*"
  else
    echo "Missing B11 scale0.75 ResNet50 seed43 anchor for B12 fine-tuning." >&2
    FAILED_TASKS+=("B12_finetune:missing_anchor")
  fi
else
  echo "Skipping B12 second-stage fine-tuning (RUN_B12_FINETUNE=${RUN_B12_FINETUNE})."
fi

if [[ "${RUN_B11_ENSEMBLE}" == "1" ]]; then
  # Exact, pre-declared valid ensemble. Never point this block at the locked test.
  eval_b_ensemble "B_balanced_v2_valid_B11_top3_ensemble_thr0p45" "0.45" \
    "B11_scale075_resnet50_cls2w105_os20_seed43_*" \
    "B10_resnet50_cls2w105_os20_seed44_*" \
    "B11_fullres_resnet34_cls2w105_os20_seed44_*"
  eval_b_ensemble "B_balanced_v2_valid_B11_top3_ensemble_thr0p50" "0.50" \
    "B11_scale075_resnet50_cls2w105_os20_seed43_*" \
    "B10_resnet50_cls2w105_os20_seed44_*" \
    "B11_fullres_resnet34_cls2w105_os20_seed44_*"
else
  echo "Skipping B11 ensemble evaluation (RUN_B11_ENSEMBLE=${RUN_B11_ENSEMBLE})."
fi

if [[ "${RUN_B14_HALFDAY}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B14 focused half-day queue after B12/B13.
  #
  # Evidence used:
  # - ROI512/p=0.20 improved the same-seed scale=0.75 anchor by ~0.0105.
  # - scale=0.875 improved the ResNet50 multi-seed mean by ~0.005.
  # - short warm-start fine-tuning improved the old B11 anchor within 6 epochs.
  # This queue exploits those signals and does not revisit failed decoders,
  # small ROI crops, low LR=1e-5, or aggressive class weighting.
  # -------------------------------------------------------------------------

  # Direction A (highest upside): combine scale=0.875 with ROI512/p=0.20.
  for seed in 42 43 45; do
    run_train "B14_r50_s0875_mixroi512_p020_seed${seed}" \
      "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" "${seed}" 8 \
      --scale --scale-factor 0.875 \
      --model-architecture smp_unet --encoder-name resnet50 \
      --class-weights 1 1 1.05 1 \
      --oversample-class-indices 2 --oversample-factor 2.0 \
      --train-patch-size 512 \
      --patch-sampling-probability 0.20 --patch-resize-to-full \
      --patch-positive-probability 0.90 --patch-class-indices 2 \
      --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
      --no-test-after-training
  done

  # Direction B: replicate the current ROI512 leader across independent seeds.
  for seed in 42 43 44; do
    run_train "B14_r50_s075_mixroi512_p020_seed${seed}" \
      "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" "${seed}" 10 \
      --scale --scale-factor 0.75 \
      --model-architecture smp_unet --encoder-name resnet50 \
      --class-weights 1 1 1.05 1 \
      --oversample-class-indices 2 --oversample-factor 2.0 \
      --train-patch-size 512 \
      --patch-sampling-probability 0.20 --patch-resize-to-full \
      --patch-positive-probability 0.90 --patch-class-indices 2 \
      --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
      --no-test-after-training
  done

  # Direction C: short stage-2 fine-tuning from the B13 ROI512 leader.
  if B13_ROI512_ANCHOR="$(latest_run_dir 'B13_r50_s075_mixroi512_p020_seed45_*')"; then
    B13_ROI512_CHECKPOINT="${B13_ROI512_ANCHOR}/models/best_model.pth"

    run_train "B14ft_roi512_p010_lr3e6_seed48" \
      "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 48 10 \
      --scale --scale-factor 0.75 \
      --model-architecture smp_unet --encoder-name resnet50 --encoder-weights none \
      --init-checkpoint "${B13_ROI512_CHECKPOINT}" \
      --epochs 40 --learning-rate 3e-6 --lr-cosine-t-max 30 \
      --early-stopping-min-epochs 10 --early-stopping-patience 12 \
      --augmentation-curriculum none \
      --class-weights 1 1 1.05 1 \
      --oversample-class-indices 2 --oversample-factor 2.0 \
      --train-patch-size 512 \
      --patch-sampling-probability 0.10 --patch-resize-to-full \
      --patch-positive-probability 0.90 --patch-class-indices 2 \
      --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
      --no-test-after-training

    run_train "B14ft_roi512_p015_lr3e6_seed49" \
      "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 49 10 \
      --scale --scale-factor 0.75 \
      --model-architecture smp_unet --encoder-name resnet50 --encoder-weights none \
      --init-checkpoint "${B13_ROI512_CHECKPOINT}" \
      --epochs 40 --learning-rate 3e-6 --lr-cosine-t-max 30 \
      --early-stopping-min-epochs 10 --early-stopping-patience 12 \
      --augmentation-curriculum none \
      --class-weights 1 1 1.05 1 \
      --oversample-class-indices 2 --oversample-factor 2.0 \
      --train-patch-size 512 \
      --patch-sampling-probability 0.15 --patch-resize-to-full \
      --patch-positive-probability 0.90 --patch-class-indices 2 \
      --patch-min-positive-pixels 64 --patch-center-jitter 0.15 \
      --no-test-after-training
  else
    echo "Missing B13 ROI512 seed45 anchor; skipping the two B14 fine-tuning jobs." >&2
    FAILED_TASKS+=("B14_finetune:missing_B13_ROI512_anchor")
  fi

  # Fine valid-only threshold calibration and an explicit no-TTA comparison.
  eval_fine_valid_sweep "B14" "B14_r50_*"
  eval_fine_valid_sweep "B14ft" "B14ft_*"
else
  echo "Skipping B14 focused half-day queue (RUN_B14_HALFDAY=${RUN_B14_HALFDAY})."
fi

if [[ "${RUN_B15}" == "1" ]]; then
  # -------------------------------------------------------------------------
  # B15: class-2 structural loss experiments on the stable scale=0.875
  # ResNet50 recipe. No ROI is used. Each new loss is isolated before the
  # combined run, then the combined candidate is repeated on seed43.
  # -------------------------------------------------------------------------
  B15_HARD_MANIFEST="runs/debug_eval/B15_class2_hard_train_top25.json"
  if B15_HARD_SOURCE="$(latest_run_dir 'B13_r50_s075_mixroi512_p020_seed45_*')"; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    if python scripts/build_hard_sample_manifest.py \
      --run-dir "${B15_HARD_SOURCE}" \
      --img-dir "${DATA_B}/train/images" --mask-dir "${DATA_B}/train/masks" \
      --output-json "${B15_HARD_MANIFEST}" \
      --class-index 2 --top-fraction 0.25 --threshold 0.45 --tta flips \
      --device cuda --num-workers 4; then
      echo "Finished B15 hard-sample mining."
    else
      rc=$?
      FAILED_TASKS+=("B15_hard_manifest:exit_${rc}")
    fi
  else
    echo "Missing B13 ROI512 source for hard-sample mining." >&2
    FAILED_TASKS+=("B15_hard_manifest:missing_source")
  fi

  run_train "B15_s0875_tversky005_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --scale --scale-factor 0.875 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 2.0 \
    --tversky-weight 0.05 --tversky-class-index 2 --tversky-alpha 0.3 --tversky-beta 0.7 \
    --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training

  run_train "B15_s0875_tversky010_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --scale --scale-factor 0.875 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 2.0 \
    --tversky-weight 0.10 --tversky-class-index 2 --tversky-alpha 0.3 --tversky-beta 0.7 \
    --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training

  run_train "B15_s0875_boundary005_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --scale --scale-factor 0.875 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 2.0 \
    --boundary-loss-weight 0.05 --boundary-class-index 2 --boundary-kernel-size 3 \
    --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training

  run_train "B15_s0875_boundary010_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
    --scale --scale-factor 0.875 --encoder-name resnet50 \
    --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 2.0 \
    --boundary-loss-weight 0.10 --boundary-class-index 2 --boundary-kernel-size 3 \
    --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training

  for seed in 42 43; do
    run_train "B15_s0875_tv005_bd005_seed${seed}" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" "${seed}" 8 \
      --scale --scale-factor 0.875 --encoder-name resnet50 \
      --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 2.0 \
      --tversky-weight 0.05 --tversky-class-index 2 --tversky-alpha 0.3 --tversky-beta 0.7 \
      --boundary-loss-weight 0.05 --boundary-class-index 2 --boundary-kernel-size 3 \
      --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training
  done

  if [[ -f "${B15_HARD_MANIFEST}" ]]; then
    run_train "B15_s0875_hard15_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
      --scale --scale-factor 0.875 --encoder-name resnet50 \
      --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 1.0 \
      --hard-sample-manifest "${B15_HARD_MANIFEST}" --hard-sample-factor 1.5 \
      --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training

    run_train "B15_s0875_hard15_tv005_bd005_seed42" "${DATA_B}" "${DATA_B_CURATED}" "${DATA_B_CURATED}" 42 8 \
      --scale --scale-factor 0.875 --encoder-name resnet50 \
      --class-weights 1 1 1.05 1 --oversample-class-indices 2 --oversample-factor 1.0 \
      --hard-sample-manifest "${B15_HARD_MANIFEST}" --hard-sample-factor 1.5 \
      --tversky-weight 0.05 --tversky-class-index 2 --tversky-alpha 0.3 --tversky-beta 0.7 \
      --boundary-loss-weight 0.05 --boundary-class-index 2 --boundary-kernel-size 3 \
      --early-stopping-min-epochs 60 --early-stopping-patience 20 --no-test-after-training
  else
    echo "Hard-sample manifest unavailable; skipping the two B15 hard-mining runs." >&2
  fi

  eval_fine_valid_sweep "B15" "B15_*"

  # Existing complementary models: class-2 ROI leader, high-C0/C3 LR3e-5,
  # and the stable scale=0.875 model. Evaluate class-specific model weights,
  # class-specific thresholds, and absolute multi-scale TTA on valid only.
  if B15_ROI="$(latest_run_dir 'B13_r50_s075_mixroi512_p020_seed45_*')" && \
     B15_LR3="$(latest_run_dir 'B13_r50_s075_lr3e5_seed45_*')" && \
     B15_S0875="$(latest_run_dir 'B12_r50_s0875_seed43_*')"; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    if python scripts/evaluate_ensemble_on_split.py \
      --run-dir "${B15_ROI}" --run-dir "${B15_LR3}" --run-dir "${B15_S0875}" \
      --class-model-weights "0.20,0.20,0.60,0.20" \
      --class-model-weights "0.50,0.30,0.15,0.50" \
      --class-model-weights "0.30,0.50,0.25,0.30" \
      --thresholds 0.40 0.50 0.425 0.50 \
      --inference-scales 0.75 0.875 1.0 \
      --evaluation-scale 0.875 --tta flips \
      --img-dir "${DATA_B_CURATED}/valid/images" \
      --mask-dir "${DATA_B_CURATED}/valid/masks" \
      --output-json "runs/debug_eval/B15_valid_classweighted_multiscale_ensemble.json" \
      --device cuda --batch-size 2 --num-workers 4; then
      echo "Finished B15 class-weighted multi-scale ensemble."
    else
      FAILED_TASKS+=("B15_advanced_ensemble:exit_$?")
    fi
  else
    echo "Missing one or more B13/B12 ensemble anchors; skipping B15 advanced ensemble." >&2
    FAILED_TASKS+=("B15_advanced_ensemble:missing_anchor")
  fi
else
  echo "Skipping B15 structural-loss queue (RUN_B15=${RUN_B15})."
fi

if [[ "${RUN_REFERENCE_EVAL}" == "1" ]]; then
  if [[ "${RUN_B6}" == "1" ]]; then
    eval_b_runs "B_original_test_tta_reference_B6" "${DATA_B}" "flips" "B6_*"
  fi
  if [[ "${RUN_B7}" == "1" ]]; then
    eval_b_runs "B_original_test_tta_reference_B7" "${DATA_B}" "flips" "B7_*"
  fi
  if [[ "${RUN_ROI_PATCH}" == "1" ]]; then
    eval_b_runs "B_original_test_tta_reference_B8_roi" "${DATA_B}" "flips" "B8_*"
  fi
  if [[ "${RUN_NIGHT}" == "1" ]]; then
    eval_b_runs "B_original_test_tta_reference_B5" "${DATA_B}" "flips" "B5_*"
  fi
else
  echo "Skipping raw B reference evaluation (RUN_REFERENCE_EVAL=${RUN_REFERENCE_EVAL})."
fi

echo "============================================================"
if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "Completed with ${#FAILED_TASKS[@]} failed task(s):"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  echo "Inspect runs/logs/run_B*.log and runs/debug_eval/*_B*_history_eval.csv."
  exit 1
fi

echo "All requested B-only tasks completed successfully."
echo "Primary metric: 4-class global Dice on the B curated test split."
echo "B8 ROI training logs: runs/logs/run_B8_*.log"
echo "B8 ROI run directories: runs/B8_*"
echo "B8 ROI evaluation outputs: runs/debug_eval/*_B8_*.csv"
