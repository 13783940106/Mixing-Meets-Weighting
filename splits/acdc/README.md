# ACDC split identity

The original `train.list` / `train_slices.list` / `val.list` / `test.list` are copied without reordering: 140 training volumes (1312 slices), 20 validation volumes and 40 test volumes. They correspond to 70 / 10 / 20 patients with ED/ES volumes.

The trainer takes the first 68 train slices for 5% (labelnum=3) and the first 136 for 10% (labelnum=7). The labeled/unlabeled slice and patient lists record those exact prefixes and complements. Keep the train list order. The percentages are the existing nominal patient-label settings (3/70 and 7/70), not an alternative slice-percent protocol.

The HDF5 image/label arrays use ZYX volumes and YX slices. Native spacing remains in source NIfTI headers; the provided preprocessor exports `spacing.json`. The paper ACDC evaluator uses grid distances without physical voxel spacing.
