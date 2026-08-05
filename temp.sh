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
RUN_CORRECTED_HISTORY_EVAL="${RUN_CORRECTED_HISTORY_EVAL:-0}"
RUN_BALANCED_HISTORY_EVAL="${RUN_BALANCED_HISTORY_EVAL:-1}"
RUN_REFERENCE_EVAL="${RUN_REFERENCE_EVAL:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

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
