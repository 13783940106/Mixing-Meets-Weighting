# Persistent Auto-Execution Contract

Updated: 2026-07-08

This file is a durable recovery anchor for context compaction. It must be read before resuming the PB2D innovation loop.

## Non-Negotiable Execution Rule

`NEXT_LEGAL_ACTION_EXISTS = AUTO_CONTINUE_REQUIRED`

The following states are execution cursors, not stopping points:

- `FULL_RUNNING`
- `NATURAL_COMPLETION_PENDING`
- `STATIC_AUDIT_REQUIRED`
- `SCORE_RED_FLAG_AUDIT_REQUIRED`
- `LEGALITY_GATE_REQUIRED`
- `CLOSURE_REQUIRED`
- `PREFLIGHT_REQUIRED`
- `GATE_REQUIRED`
- `SMOKE_REQUIRED`
- `FULL_VALIDATION_REQUIRED`
- `MATCHED_AUDIT_REQUIRED`
- `CROSS_DATASET_D0_REQUIRED`
- `D1_REQUIRED`
- `MULTI_SEED_REQUIRED`
- `ABLATION_REQUIRED`
- `FREEZE_REQUIRED`
- `FINAL_REPORT_REQUIRED`

Do not stop after a stage summary, candidate closure, audit completion, failed 5k gate, failed full, or idle GPU. Continue to the next legal action unless a real blocker exists.

## Official PB2D Baseline and Gates

- Official baseline: PB2D Mask Token matched Arm A OFF.
- Official baseline best validation DSC: `0.872497`.
- Official baseline final validation DSC: `0.861639`.
- Historical PB2D best `0.878682` is a historical reference only and must not be used for formal improvement calculation.
- Shared 5k Arm A best anchor: `0.852812 @3000`.
- Corrected 5k promotion threshold: candidate best validation DSC at or before 5000 must be strictly greater than `0.855812`.
- Full promotion requires a successful smoke, pre-run Arm A alignment gate, and corrected 5k gate pass.
- No PROMISE12/PB2D test access.

## Candidate Loop

Each candidate must follow:

`D0 four-source audit -> candidate contract -> preflight alignment -> minimal default-off patch -> compile -> smoke -> 5k -> corrected 5k gate -> full only if authorized -> closure -> decision ledger -> next candidate`

Forbidden shortcuts:

- Do not start 5k before Arm A/pretrain/evaluator/split/seed/contract alignment.
- Do not start full before corrected 5k gate pass and full preflight.
- Do not tune weights/thresholds/delays after failure unless there is explicit defect evidence and a new mechanism card.
- Do not rebrand failed families as new candidates.

## Literature Search Rule

Prefer 2026 top-tier work. Medical imaging is preferred, but strong non-medical segmentation/structure/topology/uncertainty ideas are allowed when they map to an observed PB2D defect.

Every new candidate must record:

- 2026 literature motivation when available.
- Local source-code insertion point.
- Why it is not just weight tuning.
- Which previous failed families it avoids.
- Why it targets the observed defect.

## Current Cursor

After `CMSC_LITE_V1` failed the corrected 5k gate, continue 2026 defect-driven source-pool discovery from canonical Arm A.

