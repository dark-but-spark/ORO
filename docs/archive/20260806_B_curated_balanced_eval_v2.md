# B curated balanced evaluation v2 (2026-08-06)

Project-level handoff: see `docs/archive/20260806_PROJECT_ARCHIVE.md`.

## Why the split was changed

The previous curated split contained no class-0-positive image in `test`. Its
class-0 Dice therefore measured false positives only and could switch between
0 and 1 without measuring class-0 recall. It was not a complete four-class
evaluation set.

## Construction

- Source: `data/385-liver.groupclean.v1_curated_eval_20260802`
- Output: `data/385-liver.groupclean.v1_curated_eval_balanced_20260806`
- Only the 91 strict reviewed samples from the previous curated valid/test pool
  were used; lower-confidence class-0 cases were not silently admitted.
- Samples were assigned by original-image group (the stem before `.rf.`), so all
  augmented variants of one original remain in the same split.
- The source dataset was not modified. Files in v2 are hardlinks where supported.

## Verified distribution

| Split | Images | Original groups | Class-positive images (0/1/2/3) | Class groups (0/1/2/3) |
|---|---:|---:|---:|---:|
| valid | 47 | 17 | 3 / 18 / 12 / 14 | 1 / 7 / 4 / 5 |
| test | 44 | 16 | 6 / 17 / 9 / 12 | 2 / 7 / 3 / 4 |

Checks:

- image and mask counts match in both splits;
- all four classes are positive in both splits;
- valid/test original-group overlap is zero;
- valid/train and test/train original-group overlap is zero.

Exact assignments and pixel counts are recorded in `summary.json` and
`balanced_manifest.csv` inside the v2 dataset.

## How scores must be interpreted

Scores on v1 and v2 are not directly comparable. Re-evaluate all B4-B8
checkpoints on v2 with one fixed threshold before deciding the next recipe.
Threshold selection belongs on `valid`; do not search thresholds on `test`.

This repairs class completeness but does not create a statistically strong or
fully untouched final test set. There are only three strict independent class-0
groups, leaving one in valid and two in test. In addition, the v2 pool consists
of samples already used in the earlier validation/test workflow. A robust final
claim needs newly reviewed, previously unused B originals, especially class 0.

## Immediate run

From `~/ORO/MultiResUNet`:

```bash
bash ../temp.sh
```

The default now performs evaluation only: B4-B8 checkpoints are evaluated on
balanced-v2 test with and without flip TTA, and on balanced-v2 valid at threshold
0.5. Training queues remain disabled unless their `RUN_*` switch is explicitly
enabled.
