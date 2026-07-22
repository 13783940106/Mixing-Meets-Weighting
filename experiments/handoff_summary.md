## V13.7/V13.8 final handoff — 2026-07-18T18:27:09+08:00

- Fixed parent/threshold: `Mask Token + AdaMix 0.887704` / `0.897704`.
- V13.7 result: `NO_CANDIDATE_ELIGIBLE` after targeted BSCE, FPRC, FM-SBD, and TSFPR controlled audits.
- V13.8 result: `V13_8_NO_ELIGIBLE_NEW_FAMILY / REAL_BLOCKER_ALL_LEGAL_EVIDENCE_EXHAUSTED`.
- Legally reconstructable new families all failed: registration `-0.056966`, VQ `-0.000289` and case-unsafe, head/decoder gradient firewalls only `4/24` and `6/24` direction improvements, shape dictionary `-0.019132`.
- Other 2025–2026 routes are unavailable under the frozen protocol because they require missing assets, alter the frozen optimizer/high-risk protocol, or rebrand failed families.
- Active source is byte-identical to frozen SHA `906c8f1955f6c39da614b06dcc27cc088d841791161f0dc0676b768e926828a7`. No active launcher/controller/training/GPU PID; no patch/smoke/full/test access.
- Preserve EMR V1/V2 permanent negatives, BSCE negative, SADG/SASD deferred-not-negative, and DPCR unknown.
- Core reports: `audit/v13_7_candidate_disambiguation_report.md`, `audit/v13_7_unique_top1_decision.md`, `audit/v13_8_new_mechanism_family_report.md`, `audit/v13_8_terminal_decision.md`.
- Next legal action requires new authority or a materially new legal information source.

## V13.6 pre-existing patch audit and exact source restore — 2026-07-18 18:10 CST

- Formal decision: `V13_6_DEFAULT_OFF_FAIL / RESTORE_FROZEN_FIXED_BASE_SOURCE`.
- The pre-restore active source SHA was `6e472876300d55cb44cea9dcaceca8edd4a08ab8b7fe7ff358cc41e44306a60c`; the archived OFF-smoke source has the same SHA, so the complete modification remains recoverable as evidence.
- Fixed CPU replay under seed 1337 and frozen checkpoint was tensor-exact for input, augmentation fixture, pairing, AdaMix mask, FFT view, both model logits, pseudo-label, losses, gradients, RNG, optimizer/EMA parameter names, checkpoint keys and validation fixture.
- Strict Default-Off still failed because the OFF loop executed 43 unguarded candidate `new_tensor(...)` allocations and the protocol/source surface was not fully flag-isolated.
- `train_promise12/train_promise12_bcp_fft.py` was restored by exact whole-file copy and now matches the frozen fixed-base SHA `906c8f1955f6c39da614b06dcc27cc088d841791161f0dc0676b768e926828a7` byte-for-byte. The unreferenced RGPA helper and full diff are retained as evidence.
- Candidate status: `V13_6_PREMATURE_IMPLEMENTATION`; no ON smoke/full is authorized. The completed OFF smoke is engineering evidence only.
- SADG/SASD: `DEFERRED_DUE_TO_MISSING_CONTEMPORANEOUS_PROVENANCE` (not negative). DPCR: `ELIGIBILITY_UNKNOWN_DUE_TO_MISSING_FIXED_PARENT_MODEL2`.
- The authorized V13.4 read-only requalification was then completed mechanically: `NO_ELIGIBLE_UNIQUE_TOP1`. BSCE remains archived `STACK_THIRD_NO_GAIN`; FM-SBD/FPRC/TSFPR remain ineligible; SADG/SASD is deferred; DPCR is unknown. No patch, smoke or GPU training is authorized, and the terminal cursor is `CURRENT_AUDIT_COMPLETE / NEXT_ACTION_NOT_AUTHORIZED`.
- Reports: `audit/v13_6_preexisting_patch_state_discovery.md`, `audit/v13_6_preexisting_patch_diff.md`, `audit/v13_6_preexisting_patch_default_off_equivalence.md`, `audit/v13_6_preexisting_patch_default_off_metrics.json`, `audit/v13_4_post_v13_6_unique_top1_requalification.md`; SHA manifest: `audit/v13_6_preexisting_patch_sha256.tsv`.

## V13.4 BSCE V1 full completion — 2026-07-18T16:41:15.211182+08:00

- Natural completion: `True`; status: `STACK_THIRD_NO_GAIN`.
- Best/final validation DSC: `0.883961 @27600` / `0.877981 @30000`.
- Case safety/mechanism/protocol/checkpoint: `FAIL` / `PASS` / `PASS` / `PASS`.
- Decision: `ARCHIVE_BSCE_V1_NO_COMBINATION_NO_RETUNING`.
- Next: `CURRENT_EXPERIMENT_COMPLETE / NEXT_ACTION_NOT_AUTHORIZED`. No retuning, same-family replay, fourth module, or unauthorized GPU launch.
- BSCE counterfactual seam-equivariance variants are now in `audit/v13_4_mechanism_exclusion_table.tsv`; all remaining V13.4 pool candidates were already audited not eligible.

# V13 EMR-AdaMix Current State

## V13.4 BSCE full active cursor — 2026-07-18 14:27 CST

- Unique Top 1: `BSCE_ELIGIBLE_TOP1`; matrix/ranking: `audit/v13_4_candidate_eligibility_matrix.tsv`, `audit/v13_4_candidate_ranking.md`.
- FM-SBD, SASD, FPRC, TSFPR and DPCR are not eligible. Historical SRA-SEB negative is preserved; BSCE has a hard non-rebranding contract.
- BSCE design/static/default-off/smoke all PASS. Frozen mechanism: hard AdaMix mask, detached source-probability composition, seam radius 2, MSE weight 0.05; no EMR/EEMS/SRA/entropy/margin/confidence.
- Direct 30K full is running: `pb2d_original_2d_combo_mask_token_adamix_bsce_v1_full_20260718_1450`.
- Launcher/lock PID `55309`; single controller PID `55582`; log `audit/bsce_v1_full_20260718_1450/console.log`; live state `audit/bsce_v1_full_20260718_1450/monitor_state.json`.
- Controller monitors every 30 s and automatically runs `audit/run_bsce_v1_full_postaudit.py` after natural process exit.
- Do not launch another GPU job/controller, do not early-stop or tune. Monitor to completion, then update registry/state from post-audit outputs.

## V13.4 active cursor — 2026-07-18 12:20 CST

- Stage: `V13_4_FIXED_BASE_RESIDUAL_RISK_AND_GENERATIVE_STRUCTURE_ELIGIBILITY_AUDIT`.
- Fixed parent: `Mask Token + AdaMix = 0.887704`; success threshold `0.897704`.
- No active training/GPU/controller process. Old V12 controller PID/heartbeat and zero-byte lock are stale and must not be resumed.
- EMR V1/V2 remain permanent negative; `ABANDON_ADAMIX_EEMS_FUSION` remains active. Parallel EEMS `0.889685` is historical auxiliary only.
- Current legal work is read-only fixed-base residual risk, FM-SBD/SASD and BSCE/FPRC/TSFPR/DPCR eligibility, plus primary-source literature refresh.
- Patch, smoke, short screening and full are forbidden until a unique Top 1 and all design/five-pass gates complete.
- Discovery report: `audit/v13_4_current_state_discovery.md`.

Updated: 2026-07-18T12:10:34+08:00

- Current stage: `ABANDON_ADAMIX_EEMS_FUSION_RESTORE_MASK_TOKEN_ADAMIX_REENTER_DIFFERENT_FAMILY_X_SEARCH`.
- Completed run: `pb2d_original_2d_emr_adamix_v1_full_20260718`; no active PID/GPU PID/session.
- Log: `audit/emr_adamix_v1_full_20260718/console.log`.
- Parent: AdaMix `0.887319`; KEEP `0.890319`; STRONG `0.892319`.
- Result: best `0.885899 @23200`, final `0.881132`, delta vs parent `-0.001420`.
- Case-safety `FAIL_SYSTEMATIC_HARM`; mechanism endpoint `PASS`; decision `EMR_ADAMIX_NEGATIVE`.
- V1 identity is permanently locked and must not be deleted, overwritten, rewritten, tuned, or combined with Mask Token.
- V13.1 read-only diagnosis decision: `EMR_ADAMIX_V2_DESIGN_ELIGIBLE`.
- Diagnosis: V1 post-mix downweighting suppresses difficult correct foreground/boundary; local donor quality remains distinguishable, but R_EM is binary-confidence-equivalent and raw routing risks background collapse.
- V2 minimal patch, all static/default-off/fail-closed gates, GPU smoke and the five-pass full prelaunch audit are PASS.
- Mask Token OFF; standalone EEMS OFF; validation-only selection; test logged only, never a gate.
- V12 stack-third controller is stale and must not be resumed.
- Independent V2 full completed naturally: best `0.878114@27600`, final `0.870462`, delta vs AdaMix parent `-0.009205`. Protocol/checkpoint/mechanism PASS; case safety FAIL_SYSTEMATIC_HARM (1 positive, 4 negative; Case36 `-0.035788`). Decision `EMR_ADAMIX_V2_NEGATIVE`.
- Mask Token + V2 is prohibited. Fusion route is `ABANDON_ADAMIX_EEMS_FUSION`; formal base restored to `Mask Token + AdaMix = 0.887704`; parallel EEMS `0.889685` remains historical auxiliary only.
- Next legal stage is a newly gated, different-family third-innovation search (BSCE/FPRC/TSFPR/DPCR priority), excluding entropy/margin/confidence weighting and V1/V2 routing variants. No active GPU job.

## Archived V12 State Below

# Current PB2D Stack Third Controller State

Updated: 2026-07-12T01:39:39+08:00

- Fixed base stack: `Mask Token + AdaMix`, best val `0.887704`.
- Third-module accept threshold: `0.897704`.
- LGDC has completed and is archived as `STACK_THIRD_NO_GAIN`, best val `0.874780 @27400`.
- No active training process is currently verified.
- Next required action: `CONTINUE_EARLY_ACTIVE_STACK_THIRD_MODULE_SEARCH_NEXT_ROUND`.
- Controller must use a real candidate queue/manifest before launching another full.


## STACK_EARLY_PROBABILITY_HISTOGRAM_CONSISTENCY_V1 completion 20260712_014000

- EPHC full completed naturally at iter 30000.
- Best val DSC: 0.881026 @ iter 27600; fixed base: 0.887704; decision: STACK_THIRD_NO_GAIN.
- Final val/test reference: 0.872216/0.826277.
- No NaN/OOM/Traceback/non-finite signal detected; best/final checkpoints present.
- Next required action: continue early-active stack third-module search with a new non-rebranded candidate.


## STACK_EARLY_SPATIAL_GRADIENT_ALIGNMENT_V1 completion 20260712_035300

- SGA full completed naturally at iter 30000.
- Best val DSC: 0.880999 @ iter 11800; fixed base: 0.887704; decision: STACK_THIRD_NO_GAIN.
- Final val/test reference: 0.873624/0.826318.
- No NaN/OOM/Traceback/non-finite signal detected; best/final checkpoints present.
- Next required action: continue early-active stack third-module search with a new non-rebranded candidate.


## STACK_EARLY_FOREGROUND_RANK_CORRELATION_V1 completion 20260712_060200

- FRC full completed naturally at iter 30000.
- Best val DSC: 0.858471 @ iter 18400; fixed base: 0.887704; decision: STACK_THIRD_NO_GAIN.
- Final val/test reference: 0.823188/0.810312.
- No NaN/OOM/Traceback/non-finite signal detected; best/final checkpoints present.
- Next required action: continue early-active stack third-module search with a new non-rebranded candidate.
- Foreground execution requirement: when monitoring a running full, print live progress in the chat; do not only say it is running in background.


## PAC Full Completion Pause - 2026-07-12 12:32:58

- Candidate: STACK_EARLY_PATCH_AFFINITY_CONSISTENCY_V1
- Completed full at self_iter=30000.
- Best val DSC: 0.866000 @ iter 8800; test-at-best reference: 0.824781.
- Final val DSC: 0.832559 @ iter 30000; final test reference: 0.799835.
- Decision: STACK_THIRD_NO_GAIN.
- User requested pause after audit; do not launch next candidate until explicit resume.
- Audit files: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/audit/stack_early_patch_affinity_consistency_v1_stack_third_full_metrics_summary.json`, `/home/renguanming/jupyterlab/rgm/beta-FFT-main/audit/stack_early_patch_affinity_consistency_v1_stack_third_full_completion_report.md`, `/home/renguanming/jupyterlab/rgm/beta-FFT-main/audit/stack_early_patch_affinity_consistency_v1_stack_third_full_decision.md`.
## V13.5 SADG diagnosis completion — 2026-07-18T17:06:45+08:00

- Fact recovery: `PASS_WITH_CONTEMPORANEOUS_TEACHER_PEER_ARCHIVE_GAP`.
- Validation-only replay completed for the fixed parent, BSCE V1, EMR V1 and EMR V2 model1 checkpoints; no test metric, training, backward, optimizer, EMA update, or checkpoint write was used.
- The actual training-time EMA teacher and peer model2 were never archived. Initialization-proxy disagreement shows student-side information beyond proxy confidence (`AUROC 0.7205 vs 0.5975`) but covers only `38.80%` of negative-pair harmed pixels, and high-confidence wrong proxy covers `8.59%`.
- Formal decision: `SADG_HYPOTHESIS_NOT_SUPPORTED`.
- Terminal state: `V13_5_DIAGNOSIS_COMPLETE / SADG_NOT_ELIGIBLE`.
- No design, patch, smoke, short, full, combination, SADG V2, or other candidate is authorized.
- Evidence: `audit/v13_5_sadg_fact_recovery.md`, `audit/v13_5_sadg_negative_case_diagnosis.md`, `audit/v13_5_sadg_counterfactual_replay.json`.
