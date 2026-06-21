#!/bin/bash
set -euo pipefail

# Run from the MultiResUNet directory on the server:
#   cd ~/zjm/ORO1/ORO/MultiResUNet
#   bash ../temp.sh
#
# Current best line:
#   P_smp_resnet34_cls2w125_tta      best val_dice = 0.76646
#   P_smp_resnet34_cls2w125_os15     best val_dice = 0.76493
#
# Disk control:
#   --tb-image-interval 0 disables TensorBoard image panels.
#   --checkpoint-interval 0 disables model_epoch_N.pth periodic checkpoints.
#   --save-model still keeps best_model.pth and final model.pth.
#
# If only one GPU is available, run these one by one.

COMMON="--model-architecture smp_unet --encoder-name resnet34 --encoder-weights imagenet --validation-split 0.1 --scale --scale-factor 0.75 --input-channels 3 --output-channels 4 --batch-size 16 --learning-rate 2e-5 --gradient-clip 0.5 --weight-decay 5e-4 --num-workers 4 --prefetch-factor 2 --repeat-factor 1 --train-augmentation --augmentation-strength mild --augmentation-curriculum cosine --curriculum-start-epoch 30 --curriculum-ramp-epochs 30 --curriculum-max-aug-level 0.4 --curriculum-target-strength moderate --verbose --save-model --use-combined-loss --bce-weight 0.7 --dice-weight 0.3 --lr-scheduler cosine --device cuda --seed 42 --tb-image-interval 0 --tb-num-images 0 --checkpoint-interval 0"

# 1) Combine the two proven wins: oversampling + TTA.
nohup python train.py $COMMON --epochs 110 --early-stopping-min-epochs 70 --early-stopping-patience 25 --save-dir models/P_smp_resnet34_cls2w125_os15_tta --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w125_os15_tta --class-weights 1 1 1.25 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 --val-tta flips > run_P_smp_resnet34_cls2w125_os15_tta.log 2>&1 &

# 2) Test whether class2 oversampling 2.0 beats the current 1.5.
nohup python train.py $COMMON --epochs 110 --early-stopping-min-epochs 70 --early-stopping-patience 25 --save-dir models/P_smp_resnet34_cls2w125_os20 --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w125_os20 --class-weights 1 1 1.25 1 --oversample-class-indices 2 --oversample-factor 2.0 --oversample-min-pixels 1 > run_P_smp_resnet34_cls2w125_os20.log 2>&1 &

# 3) With oversampling enabled, test whether class2 loss weight can drop to neutral.
nohup python train.py $COMMON --epochs 110 --early-stopping-min-epochs 70 --early-stopping-patience 25 --save-dir models/P_smp_resnet34_cls2w10_os15 --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w10_os15 --class-weights 1 1 1.0 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 > run_P_smp_resnet34_cls2w10_os15.log 2>&1 &

# 4) Test whether stronger class2 loss weight stacks with os15 or starts over-penalizing.
nohup python train.py $COMMON --epochs 110 --early-stopping-min-epochs 70 --early-stopping-patience 25 --save-dir models/P_smp_resnet34_cls2w15_os15 --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w15_os15 --class-weights 1 1 1.5 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 > run_P_smp_resnet34_cls2w15_os15.log 2>&1 &

# 5) Long run for os15, but keep cosine schedule equivalent to the 100-epoch run.
nohup python train.py $COMMON --epochs 140 --lr-cosine-t-max 100 --early-stopping-min-epochs 90 --early-stopping-patience 35 --save-dir models/P_smp_resnet34_cls2w125_os15_long140_tmax100 --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w125_os15_long140_tmax100 --class-weights 1 1 1.25 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 > run_P_smp_resnet34_cls2w125_os15_long140_tmax100.log 2>&1 &

# 6) Same as current os15, but stronger L2 regularization to reduce train-val gap.
nohup python train.py $COMMON --epochs 110 --early-stopping-min-epochs 70 --early-stopping-patience 25 --weight-decay 1e-3 --save-dir models/P_smp_resnet34_cls2w125_os15_wd1e3 --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w125_os15_wd1e3 --class-weights 1 1 1.25 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 > run_P_smp_resnet34_cls2w125_os15_wd1e3.log 2>&1 &

# 7) SMP seems more robust than MultiResUNet; test slightly stronger augmentation.
nohup python train.py $COMMON --epochs 110 --early-stopping-min-epochs 70 --early-stopping-patience 25 --curriculum-max-aug-level 0.5 --save-dir models/P_smp_resnet34_cls2w125_os15_aug05 --tensorboard --log-dir runs/logs/P_smp_resnet34_cls2w125_os15_aug05 --class-weights 1 1 1.25 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 > run_P_smp_resnet34_cls2w125_os15_aug05.log 2>&1 &

# 8) Stronger ResNet encoder. If this wins, continue with resnet50 + os/TTA variants.
nohup python train.py --model-architecture smp_unet --encoder-name resnet50 --encoder-weights imagenet --validation-split 0.1 --scale --scale-factor 0.75 --input-channels 3 --output-channels 4 --epochs 110 --batch-size 12 --learning-rate 2e-5 --gradient-clip 0.5 --weight-decay 5e-4 --num-workers 4 --prefetch-factor 2 --repeat-factor 1 --train-augmentation --augmentation-strength mild --augmentation-curriculum cosine --curriculum-start-epoch 30 --curriculum-ramp-epochs 30 --curriculum-max-aug-level 0.4 --curriculum-target-strength moderate --verbose --save-model --save-dir models/P_smp_resnet50_cls2w125_os15 --tensorboard --log-dir runs/logs/P_smp_resnet50_cls2w125_os15 --use-combined-loss --bce-weight 0.7 --dice-weight 0.3 --class-weights 1 1 1.25 1 --oversample-class-indices 2 --oversample-factor 1.5 --oversample-min-pixels 1 --lr-scheduler cosine --early-stopping-min-epochs 70 --early-stopping-patience 25 --device cuda --seed 42 --tb-image-interval 0 --tb-num-images 0 --checkpoint-interval 0 > run_P_smp_resnet50_cls2w125_os15.log 2>&1 &

wait
