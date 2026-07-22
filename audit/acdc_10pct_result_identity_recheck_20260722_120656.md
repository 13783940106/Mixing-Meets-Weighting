# ACDC 10% result identity recheck

- timestamp: `20260722_120656`
- mode: strictly read-only evidence recheck; no ACDC training/eval/test/replay launched
- purpose: distinguish 2D parent/baseline, ordinary 3ch/2.5D, fixed α=0.4 probability blend, and Mask Token + AdaMix candidate

## Conclusion table

| 记录 | 准确分数 | Val/Test | 身份 | 能否作为创新点主结果 |
|---|---:|---|---|---|
| A | 0.900922624 | validation-selected / legacy registry test reported separately | ACDC 10% 2D baseline parent, seed42 official_labelnum7 | 不能作为新创新主结果；可作父模型/谱系基线 |
| B | 0.902340 | Validation only | ACDC 10% 2D matched-control seed1337 | 可作 val-only matched-control，不可写成正式 test |
| C | 0.898431 | legacy registry/test diagnostic | ordinary 3ch early-fusion 2.5D seed42 | 是 PPT 推荐历史 ordinary 3ch 值；不是创新候选 |
| D | 0.902238 | Validation only | ordinary 3ch early-fusion 2.5D matched-control seed1337 | 可作 val-only matched-control，不可替代 0.898431 seed42 PPT值 |
| E | 0.901012692 | frozen Test observational blend | fixed parent probability blend α=0.4 | 不能作为创新点主结果，只能作观测性补充 |
| F | 0.901654618 @ 25200 | Validation only; formal test deferred | Mask Token + AdaMix candidate | 可作为 val-only candidate 完成结果；结论为 neutral，不能声称正式 test 胜出 |

## Required fixed outputs

```text
ACDC_10P_PPT_RECOMMENDED_VALUE = 0.898431
ACDC_10P_PPT_RECOMMENDED_LABEL = ordinary 3ch early-fusion 2.5D seed42 historical internal baseline (not seed1337 matched validation-only, not blend, not Mask Token + AdaMix)
ACDC_10P_MASK_TOKEN_ADAMIX_FINAL_VALUE = 0.9016546176026737
ACDC_10P_MASK_TOKEN_ADAMIX_RESULT_STATUS = COMPLETE; validation-only; formal test deferred until freeze; ACDC_MASK_ADAMIX_TRANSFER_NEUTRAL
```

## Identity notes

- `0.900923` belongs to the seed42 2D parent/baseline lineage (`runs/beta_fft_acdc_official_full_seed42_fresh_ckpt`, best iter 21200).
- `0.902340` belongs to the seed1337 2D matched-control validation-only closure, not the seed42 parent.
- `0.902238` belongs to the seed1337 ordinary 3ch early-fusion 2.5D matched-control validation-only closure.
- `~0.9010` / `0.9010126923454471` belongs to the fixed parent probability blend α=0.4 on frozen test split; it consumed test access and is explicitly observational, so no retest/tuning/reselection and no innovation main-result use.
- `0.898431` is the registry/PPT historical ordinary 3ch early-fusion 2.5D seed42 value, and must not be overwritten by the later seed1337 validation-only matched-control value.
- Mask Token + AdaMix best validation mDice is `0.9016546176026737 @ 25200`; completion is real, but status is neutral/validation-only.

## Evidence files read

- `experiments/project_state.json` exists=True
- `experiments/experiment_registry.csv` exists=True
- `experiments/acdc_mask_adamix_transfer_registry.tsv` exists=True
- `experiments/acdc_mask_adamix_transfer_state.json` exists=True
- `audit/acdc_10pct_2d_baseline_metrics.json` exists=True
- `audit/acdc_10pct_2d_baseline_provenance_decision.md` exists=True
- `audit/acdc_10pct_2d_matched_control_closure_20260629_062107/final_integrity.md` exists=True
- `audit/acdc_10pct_2d_matched_control_closure_20260629_062107/control_closure_decision.md` exists=True
- `audit/acdc_10pct_25d3ch_matched_control_closure_20260629_131423/final_integrity.md` exists=True
- `audit/acdc_10pct_mask_token_adamix_metrics.json` exists=True
- `audit/acdc_10pct_mask_token_adamix_completion_report.md` exists=True
- `audit/acdc_10pct_mask_token_adamix_decision.md` exists=True
- `audit/acdc_fixed_parent_blend_alpha0p4_test_from_pb_20260628_005039/summary.json` exists=True
- `audit/acdc_fixed_parent_blend_alpha0p4_result_audit_20260628_005203/summary.tsv` exists=True

Full TSV: `audit/acdc_10pct_result_identity_recheck_20260722_120656.tsv`

## PPT safe wording

```text
ACDC 10% 2D matched control: 90.2340% Validation mDice (seed1337 val-only matched-control; not Test).
ACDC 10% ordinary 3ch early-fusion 2.5D matched control: 90.2238% Validation mDice (seed1337 val-only; not the seed42 PPT historical value).
ACDC 10% ordinary 3ch early-fusion 2.5D historical PPT value: 89.8431% / raw 0.898431 (seed42 registry/internal baseline).
Fixed parent probability blend α=0.4: about 90.10% Test mDice / raw 0.9010126923454471, inference-level observational result only.
Mask Token + AdaMix: 90.1655% Validation mDice / raw 0.9016546176026737 @ 25200; COMPLETE but validation-only, formal test deferred, neutral decision.
```

## Mask Token + AdaMix focused answers

1. 30K Full truly completed: PASS, registry/state/monitor evidence mark COMPLETE for `runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42`.
2. Best Validation mDice produced: PASS.
3. Best value: `0.9016546176026737`.
4. Best iteration: `25200`.
5. Delta vs matched parent: `+0.0007319939221491` vs parent `0.9009226236805246`; this is neutral and not a formal test win.
6. Category/case safety: referenced safety audit exists in `audit/acdc_10pct_mask_token_adamix_case_safety.tsv`; no new replay was run in this recheck.
7. Test accessed: NO for Mask Token + AdaMix final candidate in this identity; formal test deferred until freeze.
8. Final formal decision: `ACDC_MASK_ADAMIX_TRANSFER_NEUTRAL`; validation-only completion, not a formal test-positive claim.

## Five self-checks

```text
SELF_CHECK_1 = PASS  # ISIC2018 10% was not stopped, restarted, or modified.
SELF_CHECK_2 = PASS  # No ACDC training/eval/test/replay was launched; only existing files were read.
SELF_CHECK_3 = PASS  # Baseline/parent, ordinary 2.5D, fixed blend, and Mask Token + AdaMix were separated.
SELF_CHECK_4 = PASS  # No unverified ~90.1% value was promoted to innovation final result.
SELF_CHECK_5 = PASS  # ISIC2018 supervision resumed and continued after the report.
```
