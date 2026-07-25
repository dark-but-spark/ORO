# Prediction Diagnostics Analysis - 2026-07-24

Source:

```text
runsTemp/diagnostics/best_20260724_tta
runsTemp/diagnostics/best_20260724_no_tta
```

Analyzed model:

```text
P_smp_resnet34_cls2w125_os15_tta_long140_tmax100_20260724_161215/models/best_model.pth
```

## Summary

The diagnostic run reproduces the current best validation level:

```text
TTA mean overall Dice:    0.77351
No-TTA mean overall Dice: 0.76633
TTA gain:                +0.00718
Validation samples:      795
```

TTA is useful overall, but it is not uniformly safe:

```text
Samples improved by TTA: 501
Samples worsened by TTA: 282
Mean delta: +0.00718
Largest improvement: +0.66646
Largest drop:        -0.53279
```

## Distribution

TTA validation Dice distribution:

```text
mean:   0.77351
median: 0.88050
p05:    0.05821
p10:    0.42625
p25:    0.69900
p75:    0.95170
p90:    0.97702
p95:    0.98164
best:   0.99279
```

The model is already excellent on many samples, but a small hard subset drags the mean down:

```text
Dice < 0.001: 38 samples
Dice < 0.05:  40 samples
Dice < 0.10:  44 samples
Dice < 0.20:  50 samples
Dice < 0.50:  110 samples
Dice < 0.70:  199 samples
```

This means the next large gain is unlikely to come from general hyperparameter tuning. The main target is the hard-tail samples.

## Class-Level Findings

The original per-class means in `per_class_summary.json` look very high because samples where both GT and prediction are empty score Dice=1. The more useful statistic is GT-nonempty class Dice:

```text
class_0 nonempty Dice: 0.8634, GT nonempty samples: 221
class_1 nonempty Dice: 0.7585, GT nonempty samples: 218
class_2 nonempty Dice: 0.6061, GT nonempty samples: 161
class_3 nonempty Dice: 0.8270, GT nonempty samples: 195
```

Class 2 is still the weakest class. Class 1 is the second weakest. Class 0 and class 3 are much stronger.

Aggregate TTA pixel errors:

```text
class_0: true=4,398,671 pred=4,147,307 FP=500,499  FN=751,863
class_1: true=6,969,556 pred=7,336,570 FP=1,450,889 FN=1,083,875
class_2: true=1,818,623 pred=1,499,496 FP=435,259  FN=754,386
class_3: true=5,955,422 pred=5,893,093 FP=766,700  FN=829,029
```

Class 2 is under-predicted overall: `pred < true` and FN is high.

## Worst-Case Pattern

The worst 50 samples are not normal boundary errors. They contain many complete misses:

```text
TTA Dice < 0.001: 38 samples
Worst 50 Dice range: 0.0000 to 0.1837
```

Visual inspection of the first worst panels shows mask blocks and predicted blocks often do not overlap at all. Some predictions appear visually plausible but are located differently from GT. This suggests a hard-tail dominated by one or more of:

```text
1. label ambiguity or inconsistent annotation
2. object localization misses
3. class confusion on sparse targets
4. possible generated/augmented sample variants that are visually close but label positions differ
5. threshold/TTA instability on marginal samples
```

This is not the same as ordinary overfitting. Training more epochs or changing focal loss is unlikely to solve these cases.

## TTA Finding

TTA gives a clear average improvement:

```text
0.77351 vs 0.76633, +0.00718
```

But several samples collapse under TTA. Examples:

```text
16_253_308_34_4...      TTA 0.0005 vs no-TTA 0.5333
18_217_217_55_27...     TTA 0.0000 vs no-TTA 0.5257
17_209_102_7_20...      TTA 0.0000 vs no-TTA 0.4236
```

Next improvement should test confidence-aware TTA instead of simple average TTA.

## Next Strategy

Priority 1: hard-tail review and data cleanup

```text
Review worst_cases/001-050 manually.
Mark each as:
  A. annotation wrong/inconsistent
  B. model missed obvious target
  C. target visually ambiguous
  D. TTA-only collapse
  E. threshold issue
```

Priority 2: class2-focused improvement

```text
class2 nonempty Dice is only 0.6061.
Try class2-specific hard example mining or stronger class2 sampling, but avoid simply pushing class2 weight too high.
```

Priority 3: TTA refinement

```text
Do not assume TTA is always safe.
Add an evaluation mode that compares no-TTA and TTA confidence/area consistency.
Potential rule:
  use TTA only when predicted mask area is not drastically smaller/larger than no-TTA.
```

Priority 4: code upgrades

```text
1. Add GT-nonempty per-class summary to analyze_predictions.py. Done.
2. Add threshold sweep per class.
3. Add no-TTA vs TTA disagreement report.
4. Add hard-example export list for retraining.
5. Add optional ensemble evaluation after TTA stability is understood.
```

## Decision

Do not start another broad training grid yet. The most valuable next work is:

```text
1. manually inspect the 50 worst TTA panels
2. classify error causes
3. add threshold sweep + TTA disagreement diagnostics
4. then decide between data cleanup, class2 mining, boundary loss, or TTA gating
```
