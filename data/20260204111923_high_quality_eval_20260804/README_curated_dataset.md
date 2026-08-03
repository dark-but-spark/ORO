# Curated Dataset From Manual Review

生成时间：2026-08-04T00:15:14

## 来源

- source root: `data\20260204111923`
- output root: `data\20260204111923_high_quality_eval_20260804`
- selection: `strict-eval`
- min confidence: `4`

## 选择规则

- `strict-eval`: `usable_for_valid=Y`、`suggest_exclude!=Y`、`needs_relabel!=Y`、`label_confidence_1_to_5 >= min_confidence`
- `eval`: `usable_for_valid=Y`、`suggest_exclude!=Y`
- `train-usable`: `usable_for_train=Y`、`suggest_exclude!=Y`

## 本次统计

```json
{
  "created_at": "2026-08-04T00:15:14",
  "source_root": "data\\20260204111923",
  "output_root": "data\\20260204111923_high_quality_eval_20260804",
  "review_csv": [
    "E:\\cjy\\xwechat_files\\wxid_3hpoaw1pywon22_43f2\\msg\\file\\2026-08\\high_quality_candidates.csv"
  ],
  "selection": "strict-eval",
  "min_confidence": 4,
  "review_rows": 240,
  "selected_rows": 57,
  "rejected_rows": 183,
  "split_counts": {
    "valid": 34,
    "test": 23
  },
  "class_counts": {
    "ballooning": 22,
    "fibrosis": 24,
    "inflammation": 10,
    "steatosis": 1
  },
  "reject_reasons": {
    "not_usable_for_valid": 170,
    "suggest_exclude": 8,
    "needs_relabel": 5
  },
  "copy_modes": {
    "hardlink": 171
  }
}
```

## 输出

- `valid/images`, `valid/masks`, `valid/labels`
- `test/images`, `test/masks`, `test/labels`
- `curated_manifest.csv`: 全部审查行和保留/拒绝原因
- `selected_rows.csv`: 被纳入 curated split 的行
- `rejected_rows.csv`: 被排除的行
- `manual_review_merged.csv`: 原始人工批注合并表
- `data.yaml`: 标准 YOLO/项目类别说明
