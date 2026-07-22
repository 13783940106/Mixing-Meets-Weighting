"""Immutable run-directory and contract primitives for ISIC pre-Full runs."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

COLLISION = 'ISIC2018_RUN_DIRECTORY_COLLISION_BLOCKED'
def sha_json(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
CONTRACT_SCHEMA_VERSION='ISIC2018_RUN_CONTRACT_V1'
TRAINING_SEMANTIC_FIELD_SCHEMA=('dataset','label_ratio','arm','seed','budget_optimizer_updates','design_sha256','split_sha256','training_protocol_sha256','model_architecture_sha256','input_size','normalization','pretrain','num_classes','optimizer','scheduler','micro_batch_size','gradient_accumulation','effective_batch_size','validation_schedule','checkpoint_schedule','best_checkpoint_rule','unlabeled_gt_gate','test_gate','determinism_runtime_contract_sha256','ce_backend_sha256','mask_token_enabled','adamix_enabled','candidate_only_loss_enabled')
CONTRACT_REQUIRED=(
 'contract_schema_version','contract_phase','formal_result_eligible','formal_resume_eligible','working_source_tree_sha256','dataset','label_ratio','arm','stage','seed','budget_optimizer_updates',
 'design_version','design_sha256','split_protocol','split_sha256','training_protocol_version','training_protocol_sha256',
 'formal_source_snapshot_version','formal_source_snapshot_sha256','config_sha256','environment_sha256','model_class',
 'model_architecture_parent','model_architecture_sha256','component_delta_sha256','BCP_parent_parity_sha256',
 'canonical_initialization_sha256','shared_parameter_map_sha256','randomness_plan_sha256','data_order_plan_sha256',
 'train_core_module','train_core_function','train_core_source_sha256','input_size','normalization','pretrain','num_classes',
 'optimizer','scheduler','micro_batch_size','gradient_accumulation','effective_batch_size','validation_schedule',
 'checkpoint_schedule','best_checkpoint_rule','unlabeled_gt_gate','test_gate','determinism_runtime_contract',
 'determinism_runtime_contract_sha256','ce_backend_sha256','mask_token_enabled','adamix_enabled','candidate_only_loss_enabled')
def build_run_contract(**values):
    contract={'contract_schema_version':CONTRACT_SCHEMA_VERSION,**values}
    missing=[k for k in CONTRACT_REQUIRED if k not in contract]
    if missing: raise ValueError('ISIC2018_RUN_CONTRACT_MISSING_FIELDS:'+','.join(missing))
    return contract
def build_training_semantics(values):
    missing=[k for k in TRAINING_SEMANTIC_FIELD_SCHEMA if k not in values]
    if missing: raise ValueError('ISIC2018_TRAINING_SEMANTICS_MISSING_FIELDS:'+','.join(missing))
    return {k:values[k] for k in TRAINING_SEMANTIC_FIELD_SCHEMA}
def build_run_instance(*, run_name, run_root, attempt_id, run_uuid):
    return {'run_name':run_name,'run_root':str(run_root),'attempt_id':attempt_id,'run_uuid':run_uuid}
def hash_run_contract(contract): return sha_json(contract)
def compare_run_contract(expected, actual):
    keys=sorted(set(expected)|set(actual));return [{'field':k,'expected':expected.get(k),'actual':actual.get(k),'match':expected.get(k)==actual.get(k)} for k in keys]
def validate_run_contract(expected, actual):
    diff=compare_run_contract(expected,actual)
    return all(x['match'] for x in diff),diff
def create_exclusive_run_directory(path):
    path=Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(COLLISION) from exc
    (path/'checkpoints').mkdir(exist_ok=False)
    return path
def write_once(path, text):
    path=Path(path)
    if path.exists(): raise RuntimeError(COLLISION)
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as f:f.write(text)
def verify_resume_contract(run, expected):
    p=Path(run)/'run_contract.json'
    if not p.exists(): raise RuntimeError('ISIC2018_RESUME_CONTRACT_MISSING')
    got=json.loads(p.read_text())
    actual=got.get('contract',got)
    if isinstance(expected,str):
        if got.get('sha256') != expected: raise RuntimeError('INVALIDATED_BY_CONTRACT_CHANGE')
        return got
    ok,diff=validate_run_contract(expected,actual)
    if not ok: raise RuntimeError('INVALIDATED_BY_CONTRACT_CHANGE:'+json.dumps(diff,sort_keys=True))
    return got
