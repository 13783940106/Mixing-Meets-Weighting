from train_ISIC2018.isic2018_lifecycle import build_run_contract, hash_run_contract, validate_run_contract

def sample():
 det={'CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,'ce_backend':'deterministic_cross_entropy'}
 return build_run_contract(contract_phase='PRE_FREEZE_DIAGNOSTIC',formal_result_eligible=False,formal_resume_eligible=False,working_source_tree_sha256='tree',dataset='ISIC2018',label_ratio='5p',arm='baseline',stage='resume_diagnostic',seed=1337,budget_optimizer_updates=2,design_version='V2.1',design_sha256='d',split_protocol='S',split_sha256='s',training_protocol_version='V1',training_protocol_sha256='p',formal_source_snapshot_version='V1',formal_source_snapshot_sha256='f',config_sha256='c',environment_sha256='e',model_class='UNet',model_architecture_parent='train_MSCMRSEG.networks.unet.UNet',model_architecture_sha256='m',component_delta_sha256='delta',BCP_parent_parity_sha256='bcp',canonical_initialization_sha256='init',shared_parameter_map_sha256='map',randomness_plan_sha256='rng',data_order_plan_sha256='order',train_core_module='train_ISIC2018.isic2018_training',train_core_function='main',train_core_source_sha256='src',input_size=[256,256],normalization='RGB_SCALE_ONLY_0_1',pretrain='NONE',num_classes=2,optimizer='SGD',scheduler='poly',micro_batch_size=8,gradient_accumulation=1,effective_batch_size=8,validation_schedule=[0,1],checkpoint_schedule=[1],best_checkpoint_rule='highest dice earliest tie',unlabeled_gt_gate='locked',test_gate='locked',determinism_runtime_contract=det,determinism_runtime_contract_sha256='detsha',ce_backend_sha256='cesha',mask_token_enabled=False,adamix_enabled=False,candidate_only_loss_enabled=False)
c=sample(); assert hash_run_contract(c)==hash_run_contract(dict(reversed(list(c.items()))));ok,d=validate_run_contract(c,dict(c));assert ok and all(x['match'] for x in d);other=dict(c);other['split_sha256']='changed';ok,d=validate_run_contract(c,other);assert not ok and [x for x in d if not x['match']][0]['field']=='split_sha256'
print('TEST_RUN_CONTRACT_SCHEMA_COMPLETE PASS')
print('TEST_RUN_CONTRACT_CANONICAL_SERIALIZATION PASS')
print('TEST_RUN_CONTRACT_HASH_STABLE PASS')
print('TEST_RUN_CONTRACT_EXACT_MATCH PASS')
print('TEST_RUN_CONTRACT_SINGLE_FIELD_MISMATCH PASS')
if __name__=='__main__': pass
