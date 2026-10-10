# Version notes — October 10, 2026

## Changes in 422b642

- Add the module-level NumPy import used by the evaluation entry.
- Document the two U-Net students and the pretraining optimizer state inherited during SSL.
- Provide an explicit PROMISE12 augmentation-mode selector and a separate configuration for cutout-consistent training.
- Update repository links to `Mixing-Meets-Weighting`.

## PROMISE12 configurations

| Configuration | Mode | Strong image and label | Output | Result record |
| --- | --- | --- | --- | --- |
| `configs/paper/promise20.json` | `legacy_overwrite` | Jitter the original image; retain the cutout-modified label | `outputs/promise20/` | Paper result record retained |
| `configs/followup/promise20_cutout_consistent.json` | `cutout_then_jitter` | Jitter the cutout image; retain its paired cutout label | `outputs/promise20_cutout_consistent/` | Not run; no reported score |

The paper configuration preserves the behavior of release `8fbca8f`. The original FINAL3 loader snapshot is unavailable; this mode describes release compatibility. Its strong image and label can differ in the cutout region. The alternative keeps the cutout image and label aligned and uses a separate run identity and output path.

For cutout-consistent training:

```bash
python tools/train_paper.py --config configs/followup/promise20_cutout_consistent.json
```

Direct loader/trainer calls default to `cutout_then_jitter`; the paper configuration selects `legacy_overwrite` explicitly. New PROMISE12 2D full-state checkpoints record the selected mode in their lineage metadata. ACDC and COVID use their respective augmentation paths.
