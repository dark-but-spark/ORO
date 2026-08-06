# B balanced-v2 result analysis and B9 plan (2026-08-06)

## Input results

Latest result directory:
`runsNew/debug_eval(1)`

Balanced-v2 files:

- `B_balanced_v2_test_tta_B4_B8_history_eval.csv`
- `B_balanced_v2_test_notta_B4_B8_history_eval.csv`
- `B_balanced_v2_valid_tta_B4_B8_history_eval.csv`

The remaining `B_curated_*` files in the same directory are old-label history
outputs. They are kept for traceability, but the next decision uses the
`B_balanced_v2_*` files.

## Key findings

The current balanced-v2 test leader is still a whole-image ResNet34 recipe:

| Rank | Run | Test TTA Dice | Class Dice 0/1/2/3 |
|---:|---|---:|---|
| 1 | `B6_cls2w105_os20_seed42` | 0.838671 | 0.8671 / 0.8921 / 0.7765 / 0.9039 |
| 2 | `B8_roi448_pos075_all_os20_seed42` | 0.838198 | 0.8810 / 0.8818 / 0.7687 / 0.9008 |
| 3 | `B6_cls2w115_os20_lr3e5_seed42` | 0.836708 | 0.8856 / 0.8811 / 0.7748 / 0.9093 |
| 4 | `B6_cls2w110_os20_seed42` | 0.836040 | 0.8756 / 0.8861 / 0.7860 / 0.9143 |
| 5 | `B6_cls2w115_os18_seed42` | 0.835764 | 0.8773 / 0.8959 / 0.7356 / 0.8967 |

No-TTA leader:

| Rank | Run | Test no-TTA Dice | Class Dice 0/1/2/3 |
|---:|---|---:|---|
| 1 | `B8_roi448_pos075_all_os20_seed42` | 0.830653 | 0.8695 / 0.8807 / 0.7586 / 0.8887 |
| 2 | `B6_resnet50_cls2w115_os20_seed42` | 0.826262 | 0.8433 / 0.8712 / 0.7581 / 0.9039 |
| 3 | `B6_cls2w105_os20_seed42` | 0.824786 | 0.8700 / 0.8766 / 0.7523 / 0.8979 |

Valid TTA leader at threshold 0.5:

| Rank | Run | Valid TTA Dice | Class Dice 0/1/2/3 |
|---:|---|---:|---|
| 1 | `B5_scale075_cls2w125_os20_seed44` | 0.842124 | 0.9495 / 0.8729 / 0.7416 / 0.9339 |
| 2 | `B6_resnet50_cls2w115_os20_seed42` | 0.837217 | 0.9200 / 0.8734 / 0.7510 / 0.9114 |
| 3 | `B6_anchor_cls2w115_os20_seed44` | 0.836620 | 0.9179 / 0.8772 / 0.7295 / 0.9219 |
| 4 | `B5_scale075_resnet50_os20_seed42` | 0.836121 | 0.8321 / 0.8856 / 0.7435 / 0.9138 |
| 5 | `B8_roi448_pos075_all_os20_seed42` | 0.832943 | 0.8900 / 0.8604 / 0.7485 / 0.9080 |

## Interpretation

Balanced-v2 raises the old leader from about 0.823 to about 0.839, but the
model is still far from 0.9. The ceiling is not primarily class 0 anymore in the
current table; the persistent bottleneck is class 2. Among top models, class 2
usually sits near 0.74-0.79 while classes 0/1/3 are around 0.86-0.91.

ROI/patch is now competitive but not clearly better. `B8_roi448` nearly ties the
whole-image leader and wins no-TTA, which suggests larger context plus native
detail is useful, but pure ROI is not yet a promotion candidate by itself.

More parameters are not the main immediate lever. ResNet50 is competitive but
does not beat the ResNet34 leader under TTA. Keep one modest capacity check, but
spend most runs on seed stability and class-2-focused data/loss settings.

## Next run

`temp.sh` now contains a B9 balanced-v2 queue. Recommended server command:

```bash
cd ~/ORO/MultiResUNet
RUN_B9=1 RUN_BALANCED_HISTORY_EVAL=0 bash ../temp.sh
```

B9 directions:

- reproduce `cls2w105_os20` with seeds 43/44/45;
- test `cls2w110` again because it had the best class-2 Dice among top runs;
- try `cls2w110` with BCE:Dice 0.6:0.4;
- test mild extra class-2 oversampling at 2.4;
- verify ROI448 with one new seed and one class-2-targeted variant;
- run one ResNet50 `cls2w105` capacity check.

B9 trains with `--no-test-after-training`. It evaluates B9 checkpoints on
balanced-v2 valid at thresholds 0.45, 0.50, and 0.55. After that, choose the
best recipe and threshold from valid, then run one locked balanced-v2 test
evaluation for that short list.
