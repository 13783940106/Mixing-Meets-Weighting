# Candidate Card: pb25d_v2_parent_endpoint_monitoring_v0_20260627_225054

Status: AUDIT_ONLY_PREREGISTERED_NOT_TRAINABLE

## Parent
- parent_id: prediction_disagreement_neighbor_regularization_v2
- parent_status: CURRENT_STRONGEST_VALIDATION_25D_PARENT; GLOBAL_GAIN_WITH_CASE_SAFETY_FAILURE; NOT_BRIDGE_ELIGIBLE
- parent_global_strength: best VAL_VOL_DSC 0.880049, mean delta +0.001367 vs PB2D
- parent_local_strength: Case35 and Case36 positive vs PB2D
- parent_risk_subgroup: Case37 severe under-segmentation / FN / under-volume; Case38/39 mild watch

## This v0 candidate
This is not a training candidate. It is a parent-preserving endpoint-monitoring preregistration to collect/standardize label-free endpoints before any compensating mechanism is allowed.

## Why not train yet
Endpoint audit found Case37 severe harm, but Case35 also has worse FN while remaining a v2 positive unit. Therefore a naive FN compensation path could damage retained parent strength. SUBGROUP_EVIDENCE_LIMITED remains active.
