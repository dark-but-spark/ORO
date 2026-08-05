# B Domain Training Status On 2026-08-05

## Current State

As of 2026-08-05, the B-only training results from the latest completed queue have been archived to:

```text
D:/project/ORO/runsArchive/20260805_B5_B_only_curated_eval_and_B6_plan/
```

`runsNew/` has been cleared after archive consolidation, so the next copied-back results can stay separate from this completed batch.

The active runnable script remains:

```text
D:/project/ORO/temp.sh
```

Its current default behavior is to run the B6 queue, not the historical B4/B5 queues.

## What We Know

- Final deployment target is B domain, not mixed A+B.
- A labels have more annotation noise, so A-derived peaks are not used as the B final target.
- The valid selection metric is 4-class global Dice on the curated B evaluation split.
- The current best completed B result is `B5_scale075_cls2w115_os20_seed42_20260805_023333` with curated-B TTA Dice `0.832412`.
- The corresponding no-TTA score is `0.819385`, so the winner is not just a TTA-only artifact.
- The previous `class2 weight=1.25` anchor no longer looks like the strongest center point once B is evaluated independently.

## Recommended Baseline

Use the following as the B baseline until B6 proves otherwise:

```text
model: SMP-Unet
encoder: resnet34
scale_factor: 0.75
class_weights: [1.0, 1.0, 1.15, 1.0]
class2_oversample_factor: 2.0
loss: BCE/Dice = 0.7/0.3
augmentation: mild + cosine curriculum, max level 0.4
learning_rate: 2e-5
batch_size: 16
val/test TTA: flips
```

## Why We Are Not Expanding The Model Mainline Yet

More parameters are not ruled out, but they are not the current main improvement path:

- The B5 resnet50 run reached a strong validation peak, but curated-B test Dice still stayed below the new resnet34 winner.
- The current bottleneck looks more like recipe stability and B-domain bias calibration than pure backbone capacity.
- Because seed variance is still visible, spending the next overnight budget on reproducibility plus local parameter search is a better trade than jumping directly to larger encoders.

This means model expansion stays as a probe branch, not the default branch.

## Next Steps

Run the B6 queue in `temp.sh` and use it to answer two questions first:

1. Is `class2 weight=1.15 + oversample=2.0` stable across unseen seeds?
2. Is the real improvement coming from local weight/sampling tuning, optimization tuning, augmentation tuning, or capacity?

If B6 still plateaus around the current level, then the next structural step is more likely to be architecture diversity or ensembling, not just a larger resnet encoder.
