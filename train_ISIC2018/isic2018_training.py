#!/usr/bin/env python3
"""Single, locked ISIC pre-Full train core (probe/smoke/short/full share this file)."""
from __future__ import annotations
import argparse, copy, json, random, sys, time, os, math, uuid
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from train_ISIC2018.isic2018_common import atomic_save, canonical_sha, capture_rng, dice_loss, fft_swap, parent_bcp_plan, restore_rng, sha256_file, supervised_loss, mixed_supervised_loss
from train_ISIC2018.isic2018_dataset import ISICDataset, counter_snapshot, reset_counters
from train_ISIC2018.isic2018_models import make_model
from train_ISIC2018.isic2018_candidate_modules import MaskTokenAdapter, DecoderHeadRouter, adamix_pair
from train_ISIC2018.isic2018_lifecycle import build_run_contract, build_training_semantics, build_run_instance, create_exclusive_run_directory, hash_run_contract, verify_resume_contract, write_once

SOURCE=[Path(__file__),ROOT/'train_ISIC2018/isic2018_dataset.py',ROOT/'train_ISIC2018/isic2018_models.py',ROOT/'train_ISIC2018/isic2018_common.py',ROOT/'train_ISIC2018/isic2018_candidate_modules.py']
SPLIT=Path('/home/renguanming/datasets/ISIC2018/splits/isic2018_official_5p10p_seed1337_v1')
DESIGN=ROOT/'audit/isic2018_minimal_adaptation_canonical_v2_1.json'

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',required=True,choices=['preflight','resource-probe','smoke','short','full','validate-only','resume-diagnostic']);p.add_argument('--resume');p.add_argument('--interrupt-after',type=int);p.add_argument('--dry-run',action='store_true');p.add_argument('--confirm-full',action='store_true');p.add_argument('--confirm-design-sha');p.add_argument('--confirm-protocol-sha');p.add_argument('--confirm-split-sha');p.add_argument('--confirm-source-sha');p.add_argument('--confirm-config-sha');return p.parse_args()
def state_hash(m):
 h=torch.tensor([],dtype=torch.uint8)
 import hashlib
 x=hashlib.sha256()
 for v in m.state_dict().values(): x.update(v.detach().cpu().contiguous().numpy().tobytes())
 return x.hexdigest()
def seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def determinism_contract():
 return {
  'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
  'torch_deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),
  'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
  'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),
  'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
  'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else False,
  'ce_backend':'train_ISIC2018.isic2018_common.deterministic_cross_entropy',
  'ce_backend_sha256':sha256_file(ROOT/'train_ISIC2018/isic2018_common.py'),
  'torch':torch.__version__,
  'cuda':torch.version.cuda,
 }
def order(ids,seed,step,n):
 r=random.Random(seed+step*1000003); return [ids[r.randrange(len(ids))] for _ in range(n)]
def batch(ds, ids):
 xs=[ds[ds.ids.index(i)] for i in ids];out={'image':torch.stack([x['image'] for x in xs]),'sample_id':ids}
 if 'target' in xs[0]:out['target']=torch.stack([x['target'] for x in xs])
 return out
def evaluate(model, ds, device):
 model.eval(); vals=[];ious=[];fg=[]
 with torch.no_grad():
  for start in range(0,len(ds),8):
   z=[ds[i] for i in range(start,min(start+8,len(ds)))];x=torch.stack([q['image'] for q in z]).to(device);t=torch.stack([q['target'] for q in z]).to(device);p=model(x).argmax(1)
   for a,b in zip(p,t):
    inter=(a.bool()&b.bool()).sum().item();den=a.sum().item()+b.sum().item();union=(a.bool()|b.bool()).sum().item();vals.append(1. if den==0 else 2*inter/den);ious.append(1. if union==0 else inter/union);fg.append(float(a.float().mean()))
 model.train();return {'mean_image_dice':float(np.mean(vals)),'mean_image_iou':float(np.mean(ious)),'prediction_fg_ratio':float(np.mean(fg))}
def checkpoint(run,cfg,models,opts,sched,scaler,iteration,best,best_iter,orders):
 m1,m2,teacher,t1,t2=models
 return {'checkpoint_schema_version':'ISIC2018_CHECKPOINT_V1','checkpoint_type':'periodic','format':'ISIC2018_FULL_RECOVERABLE_V1','model1':m1.state_dict(),'model2':m2.state_dict(),'teacher_state':teacher.state_dict(),'peer_state':'NOT_APPLICABLE','ema_state':teacher.state_dict(),'ema_teacher':teacher.state_dict(),'mask_token_model1':None if t1 is None else t1.state_dict(),'mask_token_model2':None if t2 is None else t2.state_dict(),'optimizer1':opts[0].state_dict(),'optimizer2':opts[1].state_dict(),'scheduler1':sched[0].state_dict(),'scheduler2':sched[1].state_dict(),'amp_scaler':scaler.state_dict(),'global_optimizer_update':iteration,'micro_step':iteration,'best_validation_dice':best,'best_iteration':best_iter,'sampler_position':iteration,'rng':capture_rng(),'data_order':orders,'run_contract':cfg['_contract'],'run_contract_sha256':cfg['_contract'],'contract_phase':cfg['_contract_data']['contract_phase'],'formal_result_eligible':cfg['_contract_data']['formal_result_eligible'],'config':cfg,'source_sha':cfg['_source_sha'],'protocol_sha':cfg.get('protocol_sha'),'split_sha':cfg['_split_sha'],'design_sha':cfg['_design_sha'],'environment':{'torch':torch.__version__,'cuda':torch.version.cuda},'test_access':counter_snapshot()}
def save_manifested_checkpoint(run,path,payload,checkpoint_type):
 payload['checkpoint_type']=checkpoint_type; sha=atomic_save(path,payload); z=torch.load(path,map_location='cpu',weights_only=False)
 if z.get('checkpoint_schema_version')!='ISIC2018_CHECKPOINT_V1' or z.get('run_contract_sha256')!=payload['run_contract_sha256']: raise RuntimeError('CHECKPOINT_SCHEMA_OR_CONTRACT_FAILURE')
 mp=run/'checkpoints/checkpoint_manifest.json'; manifest=json.loads(mp.read_text()) if mp.exists() else {'manifest_schema_version':'ISIC2018_CHECKPOINT_MANIFEST_V1','run_contract_sha256':payload['run_contract_sha256'],'contract_phase':payload['contract_phase'],'formal_source_snapshot_sha256':payload['config']['_contract_data']['formal_source_snapshot_sha256'],'checkpoints':[]}
 if manifest['run_contract_sha256']!=payload['run_contract_sha256']: raise RuntimeError('CHECKPOINT_MANIFEST_CONTRACT_MISMATCH')
 entry={'checkpoint_id':path.name,'checkpoint_type':checkpoint_type,'optimizer_update':payload['global_optimizer_update'],'micro_step':payload['micro_step'],'relative_path':path.name,'size_bytes':path.stat().st_size,'sha256':sha,'checkpoint_schema_version':z['checkpoint_schema_version'],'created_at':time.time(),'validated':True,'validation_dice':payload['best_validation_dice'],'resume_eligible':True}; manifest['checkpoints']=[e for e in manifest['checkpoints'] if e['checkpoint_id']!=path.name]+[entry]; manifest[checkpoint_type+'_checkpoint']=path.name;manifest['updated_at']=time.time();tmp=mp.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,sort_keys=True));json.loads(tmp.read_text());os.replace(tmp,mp);return sha
def main():
 a=parse();cfg=json.loads(Path(a.config).read_text());contract_cfg={k:v for k,v in cfg.items() if k not in ('run_name','run_root')};cfg_sha=canonical_sha(contract_cfg);source_sha=canonical_sha({str(x.relative_to(ROOT)):sha256_file(x) for x in SOURCE});design_sha=sha256_file(DESIGN);split_sha=sha256_file(SPLIT/'SHA256SUMS')
 if a.mode=='full':
  required=[a.confirm_full,a.confirm_design_sha==design_sha,a.confirm_protocol_sha==cfg.get('protocol_sha'),a.confirm_split_sha==split_sha,a.confirm_source_sha==source_sha,a.confirm_config_sha==cfg_sha]
  if not all(required):raise SystemExit('ISIC2018_FULL_RUN_LOCKED_NOT_AUTHORIZED')
 if a.dry_run: print(json.dumps({'dry_run':True,'run_name':cfg['run_name'],'config_sha':cfg_sha,'source_sha':source_sha}));return
 cfg.update({'_source_sha':source_sha,'_design_sha':design_sha,'_split_sha':split_sha})
 seed(int(cfg['seed']))
 phase=cfg.get('contract_phase','PRE_FREEZE_DIAGNOSTIC'); formal=phase=='FORMAL_FROZEN_SOURCE'; candidate=cfg['arm']=='candidate'; runtime_contract=determinism_contract();runtime_contract_sha=canonical_sha(runtime_contract)
 print('ISIC2018_DETERMINISM_RUNTIME_CONTRACT '+json.dumps(runtime_contract,sort_keys=True),flush=True)
 contract_data=build_run_contract(contract_phase=phase,formal_result_eligible=formal,formal_resume_eligible=formal,working_source_tree_sha256=source_sha,dataset='ISIC2018',label_ratio=cfg['label_ratio'],arm=cfg['arm'],stage=a.mode,seed=int(cfg['seed']),budget_optimizer_updates=int(cfg['budget']),design_version='ISIC2018_MINIMAL_ADAPTATION_V2_1',design_sha256=design_sha,split_protocol='ISIC2018_OFFICIAL_5P10P_SEED1337_V1',split_sha256=split_sha,training_protocol_version='ISIC2018_TRAINING_PROTOCOL_V1',training_protocol_sha256=cfg['protocol_sha'],formal_source_snapshot_version='V1' if formal else 'PENDING_NOT_FORMAL',formal_source_snapshot_sha256=cfg.get('formal_source_snapshot_sha256','PENDING_NOT_FORMAL'),config_sha256=cfg_sha,environment_sha256=canonical_sha({'torch':torch.__version__,'cuda':torch.version.cuda,'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}),model_class='train_MSCMRSEG.networks.unet.UNet',model_architecture_parent='train_MSCMRSEG.networks.unet.UNet',model_architecture_sha256=sha256_file(ROOT/'train_MSCMRSEG/networks/unet.py'),component_delta_sha256=cfg.get('component_delta_sha256','PENDING_PREFREEZE'),BCP_parent_parity_sha256=sha256_file(ROOT/'audit/isic2018_bcp_elementwise_test_results.json'),canonical_initialization_sha256='computed_after_model_build',shared_parameter_map_sha256='computed_after_model_build',randomness_plan_sha256=canonical_sha({'seed':1337,'bcp':'numpy global parent-compatible','candidate_rng':'candidate_only_path'}),data_order_plan_sha256=canonical_sha({'seed':1337,'sampler':'seed+iteration'}),train_core_module='train_ISIC2018.isic2018_training',train_core_function='main',train_core_source_sha256=sha256_file(Path(__file__)),input_size=[256,256],normalization='RGB_SCALE_ONLY_0_1',pretrain='NONE',num_classes=2,optimizer='SGD(momentum=0.9,weight_decay=0.0001)',scheduler='poly(power=0.9)',micro_batch_size=8,gradient_accumulation=1,effective_batch_size=8,validation_schedule=cfg['validation_steps'],checkpoint_schedule=[cfg['checkpoint_interval']],best_checkpoint_rule='highest MEAN_IMAGE_DICE; earliest iteration tie',unlabeled_gt_gate='locked',test_gate='locked',determinism_runtime_contract=runtime_contract,determinism_runtime_contract_sha256=runtime_contract_sha,ce_backend_sha256=runtime_contract['ce_backend_sha256'],mask_token_enabled=candidate,adamix_enabled=candidate,candidate_only_loss_enabled=candidate)
 contract_data['training_semantics']=build_training_semantics(contract_data);contract_data['training_semantics_sha256']=canonical_sha(contract_data['training_semantics'])
 if a.resume:
  prior=json.loads((Path(cfg['run_root'])/'run_contract.json').read_text());prior_instance=prior['contract'].get('run_instance') or {}
  if prior_instance.get('run_name') != cfg.get('run_name') or str(prior_instance.get('run_root')) != str(cfg.get('run_root')):
   raise RuntimeError('INVALIDATED_BY_CONTRACT_CHANGE_RUN_INSTANCE')
  contract_data['run_instance']=prior_instance;contract_data['run_instance_sha256']=prior['contract'].get('run_instance_sha256')
 else:
  contract_data['run_instance']=build_run_instance(run_name=cfg['run_name'],run_root=cfg['run_root'],attempt_id=cfg.get('attempt_id','new'),run_uuid=str(uuid.uuid4()));contract_data['run_instance_sha256']=canonical_sha(contract_data['run_instance'])
 cfg['_contract']=hash_run_contract(contract_data);cfg['_contract_data']=contract_data
 run=Path(cfg['run_root'])
 if a.resume:
  verify_resume_contract(run,cfg['_contract'])
 else:
  create_exclusive_run_directory(run)
 contract={'sha256':cfg['_contract'],'contract':contract_data,'contract_phase':phase,'formal_result_eligible':formal}
 if not a.resume:
  write_once(run/'config.json',json.dumps(cfg,indent=2,sort_keys=True));write_once(run/'run_contract.json',json.dumps(contract,indent=2,sort_keys=True));write_once(run/'run_contract_sha256',cfg['_contract']+'\n')
 reset_counters();seed(cfg['seed']);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');ratio=cfg['label_ratio'];lab=ISICDataset(SPLIT/'isic2018_split_manifest.csv',SPLIT/f'train_{ratio}_labeled_ids.txt','labeled');unlab=ISICDataset(SPLIT/'isic2018_split_manifest.csv',SPLIT/f'train_{ratio}_unlabeled_ids.txt','unlabeled');val=ISICDataset(SPLIT/'isic2018_split_manifest.csv',SPLIT/'validation_ids.txt','validation')
 m1,m2,teacher=make_model(device),make_model(device),make_model(device); base=copy.deepcopy(m1.state_dict());m2.load_state_dict(base);teacher.load_state_dict(base);teacher.requires_grad_(False);teacher.eval(); t1=t2=h1=h2=None
 if candidate:
  torch.manual_seed(cfg['seed']+991);t1=MaskTokenAdapter(16,4).to(device);t2=MaskTokenAdapter(16,4).to(device);t2.load_state_dict(t1.state_dict());h1=DecoderHeadRouter(m1,t1);h2=DecoderHeadRouter(m2,t2);h1.attach();h2.attach()
 params1=list(m1.parameters())+([] if t1 is None else list(t1.parameters()));params2=list(m2.parameters())+([] if t2 is None else list(t2.parameters()));o1=torch.optim.SGD(params1,lr=cfg['learning_rate'],momentum=.9,weight_decay=cfg['weight_decay']);o2=torch.optim.SGD(params2,lr=cfg['learning_rate'],momentum=.9,weight_decay=cfg['weight_decay']);s1=torch.optim.lr_scheduler.LambdaLR(o1,lambda q:max(0,1-q/cfg['budget'])**.9);s2=torch.optim.lr_scheduler.LambdaLR(o2,lambda q:max(0,1-q/cfg['budget'])**.9);scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda');it=0;best=-1.;best_iou=-1.;best_it=0;orders=[]
 if a.resume:
  z=torch.load(a.resume,map_location=device,weights_only=False)
  if z['run_contract'] != cfg['_contract']: raise RuntimeError('INVALIDATED_BY_CONTRACT_CHANGE')
  m1.load_state_dict(z['model1']);m2.load_state_dict(z['model2']);teacher.load_state_dict(z['ema_teacher'])
  if candidate:
   if z.get('mask_token_model1') is None or z.get('mask_token_model2') is None: raise RuntimeError('CANDIDATE_TOKEN_STATE_MISSING')
   t1.load_state_dict(z['mask_token_model1']);t2.load_state_dict(z['mask_token_model2'])
  o1.load_state_dict(z['optimizer1']);o2.load_state_dict(z['optimizer2']);s1.load_state_dict(z['scheduler1']);s2.load_state_dict(z['scheduler2']);scaler.load_state_dict(z['amp_scaler']);restore_rng(z['rng']);it=z['global_optimizer_update'];best=z['best_validation_dice'];best_iou=z.get('best_validation_iou',-1.);best_it=z['best_iteration'];orders=z['data_order']
 metrics=(run/'metrics.jsonl').open('a',buffering=1);start=time.time();
 def validate_save():
  nonlocal best,best_iou,best_it
  v=evaluate(m1,val,device);v['iteration']=it;v['kind']='validation';metrics.write(json.dumps(v)+'\n');
  if v['mean_image_dice']>best:best,best_iou,best_it=v['mean_image_dice'],v['mean_image_iou'],it;pay=checkpoint(run,cfg,(m1,m2,teacher,t1,t2),(o1,o2),(s1,s2),scaler,it,best,best_it,orders);pay['best_validation_iou']=best_iou;save_manifested_checkpoint(run,run/'checkpoints/best.ckpt',pay,'best')
 if not a.resume: validate_save()
 while it<cfg['budget']:
  lid=order(lab.ids,cfg['seed'],it,cfg['labeled_batch']);uid=order(unlab.ids,cfg['seed']+17,it,cfg['unlabeled_batch']);orders.append({'i':it+1,'l':lid,'u':uid});L=batch(lab,lid);U=batch(unlab,uid);x=L['image'].to(device);y=L['target'].to(device);u=U['image'].to(device);o1.zero_grad(set_to_none=True);o2.zero_grad(set_to_none=True)
  # Parent-faithful paired BCP layout: BCP precedes FFT.  ISIC has no frozen
  # strong-augmentation policy, so the approved adapter supplies the same RGB
  # geometry for the peer branch while retaining the parent tensor roles.
  lh=x.shape[0]//2; uh=u.shape[0]//2; la,lb=x[:lh],x[lh:];ua,ub=u[:uh],u[uh:]
  with torch.no_grad(): pseudo_a,pseudo_b=teacher(ua).argmax(1),teacher(ub).argmax(1)
  bcp=parent_bcp_plan(x,u,np.random); keep=bcp['keep'][:,0]; input_u1=bcp['input_u1']; input_l1=bcp['input_l1']
  input_u2,input_l2=input_u1.clone(),input_l1.clone()
  with torch.autocast(device.type,enabled=scaler.is_enabled()):
   out_u1,out_l1=m1(input_u1),m1(input_l1);out_u2,out_l2=m2(input_u2),m2(input_l2)
   mixed1=torch.where(keep.bool(),pseudo_a,y[:lh]); mixed2=torch.where(keep.bool(),y[lh:],pseudo_b)
   lu1,_,_=mixed_supervised_loss(out_u1,pseudo_a,y[:lh],keep,.5,1.0);ll1,_,_=mixed_supervised_loss(out_l1,y[lh:],pseudo_b,keep,1.0,.5)
   lu2,_,_=mixed_supervised_loss(out_u2,pseudo_a,y[:lh],keep,.5,1.0);ll2,_,_=mixed_supervised_loss(out_l2,y[lh:],pseudo_b,keep,1.0,.5)
   logits1,logits2=torch.cat((out_u1,out_l1)),torch.cat((out_u2,out_l2)); cross1=dice_loss(logits1,logits2.detach().argmax(1));cross2=dice_loss(logits2,logits1.detach().argmax(1))
   fft=fft_swap(torch.cat((input_u1,input_l1)),torch.cat((input_u2,input_l2)),30);fft1,fft2=m1(fft),m2(fft);fft_peer1=dice_loss(fft1,fft2.detach().argmax(1));fft_peer2=dice_loss(fft2,fft1.detach().argmax(1))
   beta=torch.from_numpy(np.random.beta(.1,.1,size=(uh,1,1,1))).float().to(device);mixed_unlabeled=ua*(1-beta)+ub*beta;licr_logits=m1(torch.cat((x,mixed_unlabeled))); supervised_licr=supervised_loss(licr_logits[:x.shape[0]],y)
   with torch.no_grad(), torch.autocast(device.type,enabled=False):
    teacher_mix=torch.softmax(teacher(ua.float()).float(),1)*(1-beta.float())+torch.softmax(teacher(ub.float()).float(),1)*beta.float()
   pred_mix=torch.softmax(licr_logits[x.shape[0]:].float(),1)
   consistency=.1*math.exp(-5*(1-min(1,it/max(1,cfg['budget'])))**2); licr=F.mse_loss(pred_mix,teacher_mix)*consistency
   base_loss=lu1+ll1+lu2+ll2;unsup=cross1+cross2+fft_peer1+fft_peer2;candidate_loss=torch.tensor(0.,device=device)
   if candidate:
    r=adamix_pair(input_u1,input_u2,pseudo_a,pseudo_a,y[:lh],y[:lh],torch.ones_like(pseudo_a,dtype=torch.float),torch.ones_like(pseudo_a,dtype=torch.float));candidate_loss=dice_loss(m1(r.image),r.first_target)
   loss=base_loss+unsup+supervised_licr+licr+candidate_loss
  if not torch.isfinite(loss):raise FloatingPointError(f'NaN at {it+1}')
  scaler.scale(loss).backward();scaler.step(o1);scaler.step(o2);scaler.update();s1.step();s2.step();it+=1
  with torch.no_grad():
   for q,p in zip(teacher.parameters(),m1.parameters()):q.mul_(.99).add_(p,alpha=.01)
  rec={'iteration':it,'loss':float(loss.detach()),'supervised_loss':float(base_loss.detach()),'unsupervised_loss':float(unsup.detach()),'candidate_only_loss':float(candidate_loss.detach()),'lr':o1.param_groups[0]['lr'],'amp_scale':scaler.get_scale(),'pseudo_label_fg_ratio':float(torch.cat((pseudo_a,pseudo_b)).float().mean()),'seconds':time.time()-start};metrics.write(json.dumps(rec)+'\n')
  if it in cfg['validation_steps']: validate_save()
  if it%cfg['checkpoint_interval']==0 or it==cfg['budget']:
   pay=checkpoint(run,cfg,(m1,m2,teacher,t1,t2),(o1,o2),(s1,s2),scaler,it,best,best_it,orders);pay['best_validation_iou']=best_iou;save_manifested_checkpoint(run,run/f'checkpoints/iter_{it:06d}.ckpt',pay,'periodic');save_manifested_checkpoint(run,run/'checkpoints/latest.ckpt',pay,'latest')
  if a.interrupt_after==it:
   metrics.close();print('DIAGNOSTIC_INTENTIONAL_INTERRUPTION',it);return
 metrics.close();pay=checkpoint(run,cfg,(m1,m2,teacher,t1,t2),(o1,o2),(s1,s2),scaler,it,best,best_it,orders);pay['best_validation_iou']=best_iou;save_manifested_checkpoint(run,run/'checkpoints/final.ckpt',pay,'final');final_validation=None
 try:
  for line in (run/'metrics.jsonl').read_text().splitlines():
   obj=json.loads(line)
   if obj.get('kind')=='validation': final_validation=obj
 except Exception: final_validation=None
 write_once(run/'result.json',json.dumps({'iteration':it,'best_dice':best,'best_iou':best_iou,'best_iteration':best_it,'final_validation':final_validation,'run_contract_sha256':cfg['_contract'],'contract_phase':phase,'formal_result_eligible':formal,'determinism_runtime_contract':runtime_contract,'determinism_runtime_contract_sha256':runtime_contract_sha,'training_semantics_sha256':contract_data['training_semantics_sha256'],'mask_token_enabled':candidate,'adamix_enabled':candidate,'candidate_only_loss_enabled':candidate,'counters':counter_snapshot(),'shared_init_hash':canonical_sha({k:v.cpu().numpy().tobytes().hex() for k,v in base.items()}),'elapsed_seconds':time.time()-start},indent=2));print('TRAIN_COMPLETE',run)
 if h1:h1.remove();h2.remove()
if __name__=='__main__':main()
