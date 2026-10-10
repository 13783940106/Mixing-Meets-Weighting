# Publication fixes — October 10, 2026

These changes address the review of release `8fbca8fe42fa698d5f1298ee3cea130801d72719`. TRC, TCSMix, CTW, the main Full identity, splits, validation selection and recorded results are retained. The PROMISE augmentation correction is a separate, unrun follow-up.

| Issue | Change |
| --- | --- |
| `tools/evaluate_paper.py` referenced `np.ndarray` before defining `np`, and its PROMISE metric function needed the same global name | Add module-level `import numpy as np`; keep metric formulas unchanged. |
| PROMISE 2D strong-view code discarded the cutout image but retained its modified label | `cutout_then_jitter` applies the existing jitter to `image_strong`, retaining its paired `label_strong`. `legacy_overwrite` explicitly retains the former behavior. |
| PROMISE/COVID configs implied a Swin second student and fresh SSL optimizer state | Document two independent U-Net students and a U-Net teacher, all constructed through `model_1`. Retain the recorded, unused `model_2` field. Document that both SSL optimizers load the selected pretraining checkpoint's `opt`, including momentum buffers. |

## PROMISE compatibility and the corrected follow-up

The original project's current `train_promise12/dataloaders/promise12.py` is byte-identical to the file published in release `8fbca8f` (SHA256 `8a732a3eafafee5f62c1ae951c2ea48c1ebcb64f2839d4b84cc2b3f22b64b3f6`). Both contain the overwritten-cutout behavior. The `PROMISE20_FINAL3` pretraining/self-training directories do not contain a separately saved `dataloaders/promise12.py`. This targeted check does not establish which dependency bytes the historical run imported and does not reopen historical import/source binding audits.

`configs/paper/promise20.json` explicitly selects `legacy_overwrite` to preserve the original published behavior. This compatibility mode retains the known image/label inconsistency when cutout triggers. Its paper scores and historical checkpoint record remain unchanged; preservation of this mode is not proof of exact historical reproduction.

For new training with the correction, use:

```bash
python tools/train_paper.py --config configs/followup/promise20_cutout_consistent.json
```

The follow-up retains the same seed, allocation, components, budgets and validation selection settings. It changes the 2D strong-view handling and uses a distinct run identity, output directory and checkpoint path. It has no historical score/checkpoint attached. It is not claimed to reproduce the published PROMISE result. The direct `Promise12` loader and shared trainer default to `cutout_then_jitter`; the paper config supplies its compatibility override explicitly. The selected mode is recorded in new PROMISE 2D full-state checkpoint lineage to prevent resuming across modes. No checkpoint was resumed or altered in this update.

ACDC's `WeakStrongAugment`, the COVID adapter and the existing non-paper 2.5D augmentation path are unchanged. The TCSMix confidence signal remains the teacher's all-pixel mean maximum class probability; its patch size remains dynamic. `R1_v2_B` is not substituted for main Full.

## Verification scope

Checks cover source compilation, module-definition loading without calling `main` or importing model dependencies, recorded command parsing, augmentation-mode wiring, local dependency/path references and the staged Git diff. No training, augmentation execution, model loading, inference, evaluation, checkpoint reselection or data/weight migration is performed. Passing these checks does not establish end-to-end runtime validation.
