# Ignore Class Analysis

Ignored classes: 2

## Summary

| Run | TTA | Original overall Dice | Macro Dice all classes | Macro Dice without ignored | Micro Dice all classes | Micro Dice without ignored |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| best_20260724_tta | flips | 0.773513 | 0.941829 | 0.949870 | 0.827125 | 0.844877 |
| best_20260724_no_tta | none | 0.766332 | 0.937561 | 0.946441 | 0.817589 | 0.836247 |

## Notes

- `Original overall Dice` comes from the diagnostic summary and keeps the original evaluation behavior.
- `Macro Dice without ignored` averages the remaining class Dice values, so empty/easy classes can make it much higher than original overall Dice.
- `Micro Dice without ignored` sums TP/FP/FN over the remaining classes before calculating Dice. This is usually the stricter and more useful estimate.
- This script only ignores the class during scoring. It does not remove slices, change labels, or retrain the model.
