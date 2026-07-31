# Manual Review Analysis

Source CSV: `runsTemp\manual_review_package_20260724\副本manual_review_sheet.csv`
Encoding used: `gb18030`

## Coverage

- Total rows: 110
- Filled rows: 49
- Unique filled images: 49

## review_group

| Value | Count |
| --- | ---: |
| 01_tta_worst | 49 |

## manual_error_type

| Value | Count |
| --- | ---: |
| A | 40 |
| B | 8 |
| D | 1 |

## gt_reliable

| Value | Count |
| --- | ---: |
| N | 37 |
| Y | 11 |
| <blank> | 1 |

## prediction_medically_plausible

| Value | Count |
| --- | ---: |
| N | 33 |
| Y | 15 |
| <blank> | 1 |

## suggest_fix_label

| Value | Count |
| --- | ---: |
| Y | 38 |
| N | 10 |
| <blank> | 1 |

## auto_hint

| Value | Count |
| --- | ---: |
| B_or_C_check_class2 | 22 |
| A_B_check_hard_miss | 15 |
| A_B_E_check_hard_miss_or_threshold | 7 |
| D_check_TTA_harm | 5 |

## Manual Error Type By Group

- 01_tta_worst: {'A': 40, 'B': 8, 'D': 1}

## GT Reliability By Group

- 01_tta_worst: {'N': 37, 'Y': 11, '<blank>': 1}

## TTA Delta

- Mean delta: -0.0393
- Median delta: 0.0000
- TTA worse count: 20
- TTA delta < -0.05: 9
- TTA delta < -0.20: 5

## Class2 Hard

- Filled class2 hard rows: 0

## Metrics By Manual Error Type

| Error | Count | TTA Dice | no-TTA Dice | TTA Delta | class2 Dice | class2 true px | class2 pred px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 40 | 0.0319 | 0.0651 | -0.0332 | 0.4162 | 4539.8250 | 1213.6750 |
| B | 8 | 0.0001 | 0.0667 | -0.0666 | 0.7500 | 2678.0000 | 8.3750 |
| D | 1 | 0.0000 | 0.0645 | -0.0645 | 1.0000 | 0.0000 | 0.0000 |
