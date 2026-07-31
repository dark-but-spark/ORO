# 2026-07-29 训练日志与结果归档

本目录是截至 2026-07-29 的训练记录快照。大体积 TensorBoard 事件、模型权重和诊断图片仍保留在原始目录：

- `E:\project\ORO\runsTemp\runsABCtest\logs`
- `E:\project\ORO\runsTemp\diagnostics`
- `E:\project\ORO\runsTemp\manual_review_package_20260724`

归档目录只保存轻量索引、文本日志、报告快照和关键 CSV，避免重复复制模型权重与图片。

## 目录内容

- `root_snapshots/`
  - 当前根目录训练文档与脚本快照，包括 `training_experiment_log.md`、`experiment_analysis.md`、`diagnostics_analysis.md`、`temp.sh` 和人工分析脚本。
- `run_logs/`
  - `runsTemp\runsABCtest\logs` 下的顶层 `run_*.log` 文本日志副本。
- `runsABCtest_logs_inventory.csv`
  - `runsABCtest/logs` 的完整清单，包含 run 目录、文本日志、文件数量、体积和修改时间。
- `runsABCtest_results_summary.csv`
  - 自动读取各 run 的 `history/summary.json` 与 `history/test_metrics.json` 后生成的指标表。
- `latest_results_focus.csv`
  - 最近 12 个 run 的快速对照表。
- `manual_review/`
  - 人工复核包、TTA/no-TTA 对比、人工错误类型统计与复核分析结果。
- `ignore_class_analysis/`
  - 忽略 class2 后的评分分析与样本表。

## 当前判断

旧 `P_*` 系列多数来自随机/混合验证集，最高验证 Dice 约 0.77，但不能直接作为最终泛化分数。

固定 A 数据集 `20260204111923` 上更可信的结果是：

- `U_A_20260204_anchor_scale075_cls2w10_20260728_223444`
  - best val Dice: 0.5665
  - test Dice: 0.5430
  - test Dice ignore class2: 0.6945
- `U_A_20260204_fullres_cls2w10_20260729_000602`
  - best val Dice: 0.5565
  - test Dice: 0.5331
  - test Dice ignore class2: 0.6735

固定 A 上 `scale_factor=0.75` 明显优于 full resolution。class2 仍是主要瓶颈，忽略 class2 后指标能显著上升。

B 原始 Roboflow 数据存在明显增强变体泄露，因此 `U_B_385liver_*` 的 0.80+ 验证/测试分数不能直接当作真实泛化能力。后续应优先看 `385-liver.groupclean.v1` 的训练和跨源测试结果。

## 后续使用建议

1. 训练计划以固定 train/valid/test 为主，旧随机验证只用于历史参考。
2. 新实验完成后，把 `history/summary.json`、`history/test_metrics.json` 和顶层 `run_*.log` 继续加入本归档结构。
3. 对比模型时优先看固定 A test、B_clean test、跨源 test，不再用旧 mixed validation 做主结论。
4. 人工复核继续集中在 class2、GT 漏标、TTA 变差样本和低 Dice worst cases。
