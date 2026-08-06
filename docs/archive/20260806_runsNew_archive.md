# runsNew archive note (2026-08-06)

## B6/B8/B9 balanced-v2 archive

Archive root:
`D:\project\ORO\runsTemp\runsNew_archive_20260806_B6_B8_B9_balanced_v2_complete`

This archive is the complete `runsNew` state moved on 2026-08-06 after the
B9 balanced-v2 analysis. Unlike the earlier lightweight B6/B8 archive, the
training result directories were moved into `runsTemp` instead of being left
in `runsNew`.

Archived content:
- 12 B6 whole-image balanced-v2 runs.
- 9 B8 ROI/patch exploration runs.
- 9 B9 balanced-v2 follow-up runs.
- `debug_eval`, `debug_eval(1)`, `logs`, and `logs(1)`.
- `ARCHIVE_NOTE.md` inside the archive directory.

After archiving:
- `D:\project\ORO\runsNew` is empty and ready for the next training batch.
- The archive directory contains 34 original entries plus the archive note.

Related result analysis:
- `D:\project\ORO\docs\archive\20260806_B9_balanced_v2_result_analysis.md`

## B10 validation sweep archive

Archive root:
`D:\project\ORO\runsTemp\runsNew_archive_20260806_B10_valid_sweep`

This second archive contains the B10 follow-up sweep generated after the
B6/B8/B9 archive was created.

Archived content:
- 6 B10 B-domain training runs covering input scale, full resolution,
  class-2 minimum-positive-pixel sampling, augmentation strength, and a
  ResNet-50 encoder comparison.
- `debug_eval` validation evaluation outputs.
- `logs` training and evaluation logs.
- `ARCHIVE_NOTE.md` inside the archive directory.

After the B10 archive:
- `D:\project\ORO\runsNew` is empty.
- The B10 archive contains 8 original entries plus the archive note.
