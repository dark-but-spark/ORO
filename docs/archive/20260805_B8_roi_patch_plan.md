# B8 Native-Resolution ROI/Patch Plan

Date: 2026-08-05

## Data Basis

The B training split contains 671 paired 640x640 images and four-channel masks.
Observed positive-pixel proportions are approximately:

| Class | Positive images | Positive pixels |
| ---: | ---: | ---: |
| 0 | 172 | 1.6601% |
| 1 | 174 | 4.0161% |
| 2 | 133 | 0.8450% |
| 3 | 192 | 4.5576% |

Class 2 is the sparsest channel. Valid and test remain the cleaned, fixed B
splits and are never cropped using ground truth.

## Implementation

Training can now use `--train-patch-size` to crop a square patch from the
native-resolution image before augmentation. With probability
`--patch-positive-probability`, the patch is centered near a randomly selected
positive pixel from an eligible mask channel. The remaining samples are
uniform random patches, preserving background and false-positive supervision.

Relevant options:

```text
--train-patch-size 384
--patch-positive-probability 0.75
--patch-class-indices 2
--patch-min-positive-pixels 32
--patch-center-jitter 0.20
--eval-batch-size 4
```

`--patch-class-indices` is optional and defaults to all four channels. Patch
sizes must be multiples of 32. If an eligible positive channel is absent, the
sampler falls back to a random patch.

Training patches and whole-image evaluation may use different spatial sizes
because the segmentation network is fully convolutional. B8 omits `--scale`,
so training patches retain native pixel detail and valid/test use full 640x640
images. A separate evaluation batch size limits full-resolution GPU memory.

## Default B8 Queue

The default `temp.sh` queue contains eight runs:

1. 384, positive probability 0.75, all classes, seed 42.
2. The same main recipe with seed 43.
3. Patch size 320.
4. Patch size 448.
5. Positive probability 0.50.
6. Positive probability 0.90.
7. Class-2-only positive targeting with probability 0.85.
8. Pure random 384 patches as the native-resolution control.

All runs keep the current B loss center (`class2 weight=1.05`) and class-2 file
oversampling factor 2.0. Validation TTA is disabled during training for speed;
the final history evaluator exports both flips-TTA and no-TTA results.

Thresholds 0.45, 0.50, and 0.55 are evaluated on cleaned validation only. The
test split is not used to select a threshold.

## Promotion Rule

First require the 384 main recipe to be competitive across seeds 42 and 43.
Then select patch size and foreground probability using validation and the
sample-weighted curated-B result. Do not expand to larger encoders until ROI
sampling itself shows a repeatable gain over the corrected whole-image
baseline.
