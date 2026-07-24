#!/bin/bash
set -euo pipefail

# Run from the MultiResUNet directory on the server:
#   cd ~/zjm/ORO1/ORO/MultiResUNet
#   bash ../temp.sh
#
# Purpose:
#   Diagnose the current best model on the same validation split used in training.
#   Outputs per-image metrics, per-class summary, worst-case panels, and best-case panels.
#
# Expected best model path from the latest run:
#   runs/P_smp_resnet34_cls2w125_os15_tta_long140_tmax100_20260724_161215/models/best_model.pth

BEST_MODEL="runs/P_smp_resnet34_cls2w125_os15_tta_long140_tmax100_20260724_161215/models/best_model.pth"

if [ ! -f "$BEST_MODEL" ]; then
  echo "ERROR: best model not found: $BEST_MODEL"
  echo "Edit BEST_MODEL in temp.sh if the server run directory has a different timestamp."
  exit 1
fi

OUT_TTA="runs/diagnostics/best_20260724_tta"
OUT_NO_TTA="runs/diagnostics/best_20260724_no_tta"

mkdir -p "$OUT_TTA" "$OUT_NO_TTA"

python analyze_predictions.py \
  --model-path "$BEST_MODEL" \
  --output-dir "$OUT_TTA" \
  --model-architecture smp_unet \
  --encoder-name resnet34 \
  --encoder-weights none \
  --validation-split 0.1 \
  --scale \
  --scale-factor 0.75 \
  --input-channels 3 \
  --output-channels 4 \
  --batch-size 8 \
  --num-workers 4 \
  --threshold 0.5 \
  --val-tta flips \
  --seed 42 \
  --device cuda \
  --worst-count 50 \
  --best-count 12 \
  > "$OUT_TTA/analyze_predictions.log" 2>&1

python analyze_predictions.py \
  --model-path "$BEST_MODEL" \
  --output-dir "$OUT_NO_TTA" \
  --model-architecture smp_unet \
  --encoder-name resnet34 \
  --encoder-weights none \
  --validation-split 0.1 \
  --scale \
  --scale-factor 0.75 \
  --input-channels 3 \
  --output-channels 4 \
  --batch-size 8 \
  --num-workers 4 \
  --threshold 0.5 \
  --val-tta none \
  --seed 42 \
  --device cuda \
  --worst-count 50 \
  --best-count 12 \
  > "$OUT_NO_TTA/analyze_predictions.log" 2>&1

echo "Diagnostics complete."
echo "TTA report:    $OUT_TTA"
echo "No-TTA report: $OUT_NO_TTA"
