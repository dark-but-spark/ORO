# Legacy Manual Review Train/Valid Filter

生成时间：2026-07-31T14:07:50

## 来源

- 旧人工审查表：`E:\project\ORO\docs\archive\2026-07-31\legacy_manual_review_input\manual_review_sheet_extracted.csv`
- 原始 A 数据：`E:\project\ORO\data\20260204111923`
- 输出数据：`E:\project\ORO\data\20260204111923_trainval_review_filtered_20260731`

## 筛选规则

只对 `train/valid` 应用人工筛选，`test` 保持原样复制以保证目录完整。

排除条件：

- `suggest_fix_label = Y`
- `gt_reliable = N`
- `manual_error_type = A`
- notes 明确包含漏标、错标、标注不典型、GT 漏标、人工标注错误、建议剔除等标签问题

保留条件：

- `manual_error_type = B` 的纯模型错误样本保留
- `manual_error_type = D` 的 TTA 问题样本保留
- 未人工审查的样本保留

## 数量

```text
review rows: 110
review unique images: 77
matched A images: 74
excluded train/valid images: 40
```

状态计数：

```json
{
  "train:unreviewed_kept": 2721,
  "train:reviewed_kept": 27,
  "train:excluded_label_problem": 34,
  "valid:unreviewed_kept": 781,
  "valid:excluded_label_problem": 6,
  "valid:reviewed_kept": 7,
  "test:test_unmodified": 400
}
```

## 使用建议

这个数据集适合作为下一轮训练输入，用于测试“去掉明显坏标签 train/valid 后，模型是否更稳”。
最终测试仍建议同时看：

1. 原始 A test
2. 2026-07-31 curated A test
3. curated A test ignore class2

不要把 test 人工审查结果反向用于 train/valid。
