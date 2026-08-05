# Balanced curated B evaluation split

Built from the strict reviewed curated pool. All variants of an original `.rf.` group stay in one split. Both valid and test contain all four classes. The source curated dataset is unchanged. See `summary.json` and `balanced_manifest.csv` for the exact assignment.

Class 0 has only three independent source groups (one in valid and two in test), so its estimate has high uncertainty. This re-split also reuses samples seen in the previous curated valid/test workflow; a final unbiased test requires newly reviewed, previously unused original groups.

Project archive and operating notes: `docs/archive/20260806_PROJECT_ARCHIVE.md`.
