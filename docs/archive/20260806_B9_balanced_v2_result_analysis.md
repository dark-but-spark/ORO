# B9 balanced-v2 result analysis (2026-08-06)

## Input

Latest result root:
`runsNew`

New B9 runs:

- `B9_cls2w105_os20_seed43_20260806_123513`
- `B9_cls2w105_os20_seed44_20260806_125201`
- `B9_cls2w105_os20_seed45_20260806_130823`
- `B9_cls2w110_os20_seed43_20260806_132538`
- `B9_cls2w110_os20_dice04_seed42_20260806_134332`
- `B9_cls2w105_os24_seed42_20260806_140414`
- `B9_roi448_pos075_all_os20_seed43_20260806_142143`
- `B9_roi448_pos085_cls2_os20_seed42_20260806_143833`
- `B9_resnet50_cls2w105_os20_seed42_20260806_145510`

Evaluation files:

- `runsNew/debug_eval(1)/B_balanced_v2_valid_tta_B9_thr0p45_history_eval.csv`
- `runsNew/debug_eval(1)/B_balanced_v2_valid_tta_B9_thr0p50_history_eval.csv`
- `runsNew/debug_eval(1)/B_balanced_v2_valid_tta_B9_thr0p55_history_eval.csv`

B9 was intentionally trained with `--no-test-after-training`, so the table below
is validation-only. Do not claim a new test winner from these results.

## Best B9 valid result per run

| Rank | Run | Best valid Dice | Threshold | Class Dice 0/1/2/3 |
|---:|---|---:|---:|---|
| 1 | `B9_cls2w105_os20_seed44` | 0.841601 | 0.50 | 0.9024 / 0.8822 / 0.7394 / 0.9304 |
| 2 | `B9_cls2w110_os20_seed43` | 0.838364 | 0.55 | 0.9214 / 0.8849 / 0.7204 / 0.9201 |
| 3 | `B9_resnet50_cls2w105_os20_seed42` | 0.837229 | 0.45 | 0.8530 / 0.8764 / 0.7638 / 0.9012 |
| 4 | `B9_cls2w105_os20_seed43` | 0.833298 | 0.45 | 0.9115 / 0.8781 / 0.7085 / 0.9243 |
| 5 | `B9_cls2w105_os20_seed45` | 0.830187 | 0.45 | 0.8895 / 0.8717 / 0.7019 / 0.9353 |
| 6 | `B9_cls2w105_os24_seed42` | 0.826505 | 0.50 | 0.8458 / 0.8706 / 0.7154 / 0.9246 |
| 7 | `B9_cls2w110_os20_dice04_seed42` | 0.824033 | 0.50 | 0.9323 / 0.8686 / 0.6996 / 0.9157 |
| 8 | `B9_roi448_pos075_all_os20_seed43` | 0.823032 | 0.45 | 0.7946 / 0.8563 / 0.7362 / 0.9218 |
| 9 | `B9_roi448_pos085_cls2_os20_seed42` | 0.815030 | 0.55 | 0.7623 / 0.8588 / 0.7488 / 0.9061 |

## Comparison to previous balanced-v2 references

Previous valid leader from B4-B8:

- `B5_scale075_cls2w125_os20_seed44`, valid Dice 0.842124 at threshold 0.50

Previous test leader from B4-B8:

- `B6_cls2w105_os20_seed42`, test TTA Dice 0.838671 at threshold 0.50

B9 has not yet produced a test-table winner. Its best valid result
(`B9_cls2w105_os20_seed44`, 0.841601) is close to but slightly below the previous
valid leader (0.842124).

## Interpretation

The `cls2w105_os20` recipe remains the most plausible mainline, but it is not
stable across seeds:

- seed43 best valid Dice: 0.833298
- seed44 best valid Dice: 0.841601
- seed45 best valid Dice: 0.830187

The spread is large enough that a single high seed should not be promoted by
itself.

Class 2 remains the main bottleneck. The best B9 valid class-2 Dice is from the
ResNet50 run (0.7638), but its global Dice is only third. ROI448 class-2
targeting improves class-2 a little relative to the ROI448 all-class seed, but
it hurts class 0/global Dice too much.

Increasing class-2 oversampling from 2.0 to 2.4 did not help. Increasing Dice
loss pressure with `cls2w110_os20_dice04` also did not help.

Model capacity is not a clean answer yet. ResNet50 has the strongest B9 class-2
valid Dice, but its global valid Dice still trails the best ResNet34 seed and is
not enough evidence to expand the network family broadly.

## Recommended next step

Run one locked balanced-v2 test evaluation for the short list selected from
valid:

- `B9_cls2w105_os20_seed44`, threshold 0.50
- `B9_cls2w110_os20_seed43`, threshold 0.55
- `B9_resnet50_cls2w105_os20_seed42`, threshold 0.45

If these do not beat `B6_cls2w105_os20_seed42` on test, keep B6 as the current
reference and stop spending many runs on this parameter neighborhood. The next
real improvement likely needs more B data/class-2 examples or a different
strategy, not a wider sweep around class weights.
