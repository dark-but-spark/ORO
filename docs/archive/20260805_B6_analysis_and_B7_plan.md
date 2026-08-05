# B6 Analysis And B7 Plan

Date: 2026-08-05

## Objective

- Final model serves B only.
- Primary selection metric is four-class global Dice on the curated B split.
- A+B mixed history and the old 0.7726 peak are not promotion references.

## B6 Provisional Results

The copied B6 evaluation used the old batch-equal aggregation and therefore
must be treated as provisional. Its leading results were:

| Run | TTA Dice | No-TTA Dice |
| --- | ---: | ---: |
| `B6_cls2w105_os20_seed42` | 0.834425 | 0.820117 |
| `B6_resnet50_cls2w115_os20_seed42` | 0.832347 | 0.828100 |
| `B6_cls2w115_os20_lr3e5_seed42` | 0.827515 | 0.814385 |
| `B6_cls2w115_os18_seed42` | 0.827432 | 0.814614 |

The historical ResNet34 1.15/2.0 recipe varied strongly by seed:

| Seed | TTA Dice |
| ---: | ---: |
| 42 (B5) | 0.832412 |
| 43 (B6) | 0.804987 |
| 44 (B6) | 0.819139 |

Mean: approximately 0.81885. A single best seed is not sufficient for recipe
promotion.

## Metric Correction

The previous evaluator averaged batch means with equal batch weight. The
curated test split has 43 images and evaluation batch size 8, so the final
three-image batch received the same weight as each full eight-image batch.

The evaluation and training metric paths now:

- weight global Dice, Jaccard, and loss by the actual number of images;
- aggregate per-class intersections and unions over the complete dataset;
- record the number of metric samples in evaluation output.

B4-B6 are re-evaluated by default before B7. The corrected tables, rather than
the provisional values above, are the new comparison reference.

## B7 Queue

The default queue in `D:/project/ORO/temp.sh` contains 12 runs:

1. Reproduce ResNet34 `class2 weight=1.05, oversample=2.0` with seeds 43, 44,
   and 45.
2. Search weights 1.00, 1.025, and 1.075 with seed 42.
3. Test oversampling 1.8 and 2.2 around weight 1.05.
4. Combine weight 1.05 with learning rate 3e-5.
5. Test ResNet50 with weight 1.05 and reproduce the ResNet50 1.15 recipe with
   seeds 43 and 44.

After training, B7 is evaluated with and without flips TTA. Thresholds 0.40,
0.45, 0.50, 0.55, and 0.60 are swept on the curated validation split only.
The test split is not used to choose the threshold.

## Promotion Rule

Promote a recipe only when it improves the corrected curated-B result and its
multi-seed mean is competitive. Use the validation-selected threshold once on
the locked test split. Prefer ResNet34 unless ResNet50 shows a repeatable gain
large enough to justify its additional inference cost.
