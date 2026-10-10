# Preserved evaluation conventions

Evaluation uses an explicitly supplied student-1 validation-selected checkpoint. No postprocessing, optimizer, backward pass or checkpoint selection is performed by `tools/evaluate_paper.py`.

| Setting | Reconstruction / aggregation | Distances and empty masks |
| --- | --- | --- |
| ACDC | Nearest-neighbor slice resize to 256×256 and prediction resize back to saved ZYX volumes; ordinary mean over cases and RV/MYO/LV | MedPy binary HD95 and directed prediction-to-GT ASD, without spacing. Both-empty: all metrics 0. One-empty: overlap 0, grid-diagonal distance penalty. |
| PROMISE12 | Direct staged 224×224 slice inference; saved case volume reconstruction; ordinary mean over ten internal test cases, binary prostate | MedPy binary HD95 and directed prediction-to-GT ASD, without spacing. Both-empty: all metrics 0. One-empty: overlap 0, grid-diagonal distance penalty. |
| COVID-19 | Stack XY slices along axis 2; ordinary patient-volume macro mean over four test patients; binary infection | Concatenated bidirectional surface distances: 95th percentile for HD95, mean for ASD. Stored canonical spacing on historical resized derived grid. Both-empty: Dice/IoU 1, distance 0. One-empty: overlap 0, distance array `[inf]`; ordinary mean retains nonfinite values. |

ACDC delegates to `tools/v22_0_acdc_test_frozen_checkpoint.py::infer_case`, `safe_metrics`, `summarize`. PROMISE's binary metric function and inference loop come from the existing `tools/v22_1_postfull_test_launcher.py::promise_run`. COVID uses `train_COVID19/covid19_volume_evaluator.py` and the same existing prediction loop. Publication wrappers configure paths and commands.

Training validation remains in existing trainers: ACDC averages foreground Dice across volumes/classes; PROMISE averages foreground case Dice; COVID selects patient-volume macro Dice. ACDC's historical log field `model*_mean_dice_test` reads validation in the published overlay and does not denote test-based selection.

Author-confirmed COVID Full HD95 is 22.79 mm (5%) and 11.88 mm (10%). Other spacing-only versions do not replace or rebind those results. Static packaging does not establish native-grid metric parity or regenerate results.
