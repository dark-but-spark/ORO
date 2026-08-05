# ORO B-domain project archive (2026-08-06)

## Archive scope

This archive closes the current B-domain evaluation cleanup step and records the
state before the next training search. The final model is intended to serve B
only. A-domain data is noisy and transfers poorly to B, so A/B mixed scores are
not used as promotion evidence.

## Current active artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Balanced B curated eval split | `data/385-liver.groupclean.v1_curated_eval_balanced_20260806` | Class-complete valid/test split for the next B-only comparison |
| Split generator | `MultiResUNet/scripts/rebalance_curated_eval_splits.py` | Reproducible, non-destructive grouped re-split from the strict reviewed curated pool |
| Run entrypoint | `temp.sh` | Default evaluation-only run for B4-B8 on the balanced v2 split |
| Split details | `docs/archive/20260806_B_curated_balanced_eval_v2.md` | Construction details, limitations, and immediate run command |
| This index | `docs/archive/20260806_PROJECT_ARCHIVE.md` | Project-level handoff and operating notes |

The older curated split is kept at
`data/385-liver.groupclean.v1_curated_eval_20260802`. It is useful for
traceability, but its test split had no class-0-positive image and should not be
used as the primary four-class test set.

## Balanced v2 evaluation split

Source pool: 91 strict reviewed samples from the previous curated valid/test
workflow, grouped by original image stem before `.rf.`.

| Split | Images | Original groups | Class-positive images (0/1/2/3) | Class groups (0/1/2/3) |
|---|---:|---:|---:|---:|
| valid | 47 | 17 | 3 / 18 / 12 / 14 | 1 / 7 / 4 / 5 |
| test | 44 | 16 | 6 / 17 / 9 / 12 | 2 / 7 / 3 / 4 |

Verified checks:

- valid/test image and mask counts match;
- all four classes are positive in both valid and test;
- valid/test original-group overlap is zero;
- valid/train and test/train original-group overlap is zero.

Important limitation: strict class 0 has only three independent groups, so the
class-0 estimate remains high variance. The v2 pool also reuses samples from the
earlier curated valid/test workflow, so it repairs class completeness but is not
a fresh blind final test. A defensible final claim, especially near 0.9 Dice,
needs newly reviewed and previously unused B originals, with extra class-0
coverage.

## Training/result state

`runsNew` currently contains B6 and B8 training outputs plus historical
evaluation CSV/JSON files under `runsNew/debug_eval`. These files are left in
place because `temp.sh` evaluates history from the `runs` tree on the training
server, and moving checkpoint directories would make retrospective evaluation
harder to reproduce.

Recorded old-curated references:

| Family | Best old-curated result | Note |
|---|---:|---|
| B4-B6 whole-image | B6 `cls2w105_os20_seed42`, TTA Dice 0.823113 | Best corrected sample-weighted reference on old curated test |
| B8 ROI/patch | B8 `roi448_pos075_all_os20_seed42`, TTA Dice 0.816654 | ROI/patch did not beat the whole-image reference on old curated test |
| B8 valid threshold 0.5 | B8 `roi384_pos085_cls2_os20_seed42`, valid Dice 0.858198 | Valid-only signal; not a test promotion score |

Do not compare these old-curated numbers directly with future balanced-v2
numbers. Use them only to understand which recipes were historically promising.

One duplicate/failed-looking run directory is present:
`runsNew/B8_roi384_pos075_all_os20_seed42_20260805_201638`. It contains only a
manifest and logs, with no normal checkpoint payload. It is retained as evidence
but should not be treated as a completed model.

## Current `temp.sh` behavior

Default command from the server:

```bash
cd ~/ORO/MultiResUNet
bash ../temp.sh
```

Default switches:

- `RUN_BALANCED_HISTORY_EVAL=1`
- `RUN_ROI_PATCH=0`
- `RUN_CORRECTED_HISTORY_EVAL=0`
- training queues disabled unless explicitly enabled

The default run evaluates existing B4-B8 checkpoints on:

- balanced-v2 test with flip TTA;
- balanced-v2 test without TTA;
- balanced-v2 valid with flip TTA at threshold 0.5.

Threshold selection belongs on `valid`. Do not tune thresholds on `test`.

## Next operating sequence

1. Run the default balanced-v2 history evaluation.
2. Compare B4-B8 on balanced-v2 valid/test separately from all old-curated
   results.
3. Select candidate recipes using balanced-v2 valid first. Use test once for the
   short list, not for repeated hyperparameter search.
4. If the target remains above 0.9, prioritize data work before model expansion:
   more reviewed B originals, more class-0 groups, and a fresh blind test.
5. Consider larger models only after the balanced-v2 table shows a stable
   architecture-related gain rather than a one-seed fluctuation.

## Verification performed

- `temp.sh` syntax checked with Git Bash.
- `MultiResUNet/scripts/rebalance_curated_eval_splits.py` syntax checked.
- `git diff --check` passed except for the existing `temp.sh` line-ending
  warning.
- The balanced-v2 dataset manifest and summary were generated and checked for
  split completeness and group isolation.
