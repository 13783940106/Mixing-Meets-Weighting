# Evaluation protocol

Evaluate the supplied student-1 validation-selected checkpoint with `tools/evaluate_paper.py`. Dice and IoU are reported as fractions in evaluation JSON and as percentages in the paper results table. Predictions are reconstructed into volumes before aggregation.

| Dataset | Reconstruction and aggregation | Surface distances | Empty-mask convention |
| --- | --- | --- | --- |
| ACDC | Nearest-neighbor slice resize to 256×256, followed by prediction resize to saved ZYX volumes; equally average volumes and RV/MYO/LV | MedPy binary HD95 and directed prediction-to-GT ASD; unit grid | Both empty: all metrics 0. One empty: overlap 0 and grid-diagonal distance penalty. |
| PROMISE12 | Direct 224×224 slice inference and case-volume reconstruction; equally average ten internal test cases for binary prostate segmentation | MedPy binary HD95 and directed prediction-to-GT ASD; unit grid | Both empty: all metrics 0. One empty: overlap 0 and grid-diagonal distance penalty. |
| COVID-19 | Stack XY slices along axis 2; equally average four test patient volumes for binary infection segmentation | Concatenate bidirectional surface distances; HD95 is the 95th percentile and ASD the mean, in mm, using stored canonical spacing on the resized derived grid | Both empty: Dice/IoU 1 and distance 0. One empty: overlap 0 and infinite distance; arithmetic aggregation retains infinity. |

## Metric entries

- ACDC: `tools/v22_0_acdc_test_frozen_checkpoint.py`, functions `infer_case`, `safe_metrics` and `summarize`.
- PROMISE12: the binary metric function and case-inference loop in `tools/evaluate_paper.py`.
- COVID-19: `train_COVID19/covid19_volume_evaluator.py` and the patient-inference loop in `tools/evaluate_paper.py`.

## Checkpoint selection

Training selects student 1 by validation Dice: volume/class mean for ACDC, case mean for PROMISE12 and patient-volume macro mean for COVID-19. Strict improvement retains the earliest iteration when scores tie. In the ACDC overlay, the log field `model*_mean_dice_test` refers to the validation loader.
