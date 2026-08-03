# Curated Dataset From Manual Review

生成时间：2026-08-02T17:35:54

## 来源

- source root: `data\385-liver.groupclean.v1`
- output root: `data\385-liver.groupclean.v1_curated_eval_20260802`
- selection: `strict-eval`
- min confidence: `4`

## 选择规则

- `strict-eval`: `usable_for_valid=Y`、`suggest_exclude!=Y`、`needs_relabel!=Y`、`label_confidence_1_to_5 >= min_confidence`
- `eval`: `usable_for_valid=Y`、`suggest_exclude!=Y`
- `train-usable`: `usable_for_train=Y`、`suggest_exclude!=Y`

## 本次统计

```json
{
  "created_at": "2026-08-02T17:35:54",
  "source_root": "data\\385-liver.groupclean.v1",
  "output_root": "data\\385-liver.groupclean.v1_curated_eval_20260802",
  "review_csv": [
    "E:\\cjy\\xwechat_files\\wxid_3hpoaw1pywon22_43f2\\msg\\file\\2026-08\\manual_review_sheet_385-liver.groupclean.v1_test.csv",
    "E:\\cjy\\xwechat_files\\wxid_3hpoaw1pywon22_43f2\\msg\\file\\2026-08\\manual_review_sheet_385-liver.groupclean.v1_valid.csv"
  ],
  "selection": "strict-eval",
  "min_confidence": 4,
  "review_rows": 167,
  "selected_rows": 91,
  "rejected_rows": 76,
  "split_counts": {
    "test": 43,
    "valid": 48
  },
  "class_counts": {
    "steatosis": 26,
    "inflammation": 21,
    "fibrosis": 35,
    "ballooning": 9
  },
  "reject_reasons": {
    "not_usable_for_valid": 67,
    "suggest_exclude": 9
  },
  "copy_modes": {
    "hardlink": 273
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
