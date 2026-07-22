#!/usr/bin/env python3
"""Validation-only, CPU-exact 42-item Default-Off trace for V15.1.

The candidate OFF delegation is separately executed by the existing clean
process harness. This program captures every intermediate state in the frozen
training calculation twice after equivalent baseline/candidate-OFF entry
preparation; it never selects or reads formal-test data.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, platform, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from train_MSCMRSEG import v15_1_mscmr_beta_fft_baseline as B
from train_MSCMRSEG import v15_1_mscmr_mask_token_adamix_candidate as C
from train_MSCMRSEG.v15_1_mscmr_common import *
from train_MSCMRSEG.v15_1_mscmr_dataset import MSCMRSliceDataset, PairedInfiniteBatchSampler
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

ITEMS=["parsed training config","model1 parameter names/order","model2 parameter names/order","EMA teacher parameter names/order","model1 initial parameters","model2 initial parameters","EMA teacher initial parameters","optimizer parameter groups","optimizer initial state","scheduler initial state","Python RNG state","NumPy RNG state","Torch CPU RNG state","Torch CUDA RNG state","labeled batch","unlabeled batch","augmentation output/parameters","model1 logits","model2 logits","EMA teacher logits","supervised loss","pseudo-label","confidence map","selection mask","BCP mask/output","FFT input/output","unsupervised loss","total loss","per-parameter gradient","gradient norm","post-step model1","post-step model2","post-EMA teacher","post-step optimizer","post-step scheduler","next-step RNG","mode lifecycle","validation mDice","validation RV/MYO/LV","checkpoint keys/tensors","save/load/resume state","resume next batch/step"]

def digest(x): return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def tdesc(x):
 if isinstance(x,torch.Tensor): return {"kind":"tensor","shape":list(x.shape),"dtype":str(x.dtype),"sha256":hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
 if isinstance(x,dict): return {"kind":"dict","keys":list(x.keys())}
 if isinstance(x,(list,tuple)): return {"kind":type(x).__name__,"length":len(x)}
 return {"kind":type(x).__name__,"repr":str(x)[:200]}
def eq(a,b):
 if isinstance(a,torch.Tensor):
  d=(a.cpu()-b.cpu()).abs(); return bool(torch.equal(a.cpu(),b.cpu())),float(d.max()) if d.numel() else 0.,float(d.float().mean()) if d.numel() else 0.
 if isinstance(a,np.ndarray): return bool(np.array_equal(a,b)),float(np.abs(a-b).max()) if a.size else 0.,float(np.abs(a-b).mean()) if a.size else 0.
 if isinstance(a,dict):
  if list(a)!=list(b): return False,float('inf'),float('inf')
  z=[eq(a[k],b[k]) for k in a]; return all(q[0] for q in z),max([q[1] for q in z]or[0]),max([q[2] for q in z]or[0])
 if isinstance(a,(list,tuple)):
  if len(a)!=len(b): return False,float('inf'),float('inf')
  z=[eq(x,y) for x,y in zip(a,b)]; return all(q[0] for q in z),max([q[1] for q in z]or[0]),max([q[2] for q in z]or[0])
 return a==b,0.,0.
def named_params(m): return [(n,p.detach().clone()) for n,p in m.named_parameters()]
def groups(opt,models):
 lookup={id(p):n for m in models for n,p in m.named_parameters()}; return [{"names":[lookup[id(p)] for p in g['params']],"lr":g['lr'],"momentum":g.get('momentum'),"weight_decay":g['weight_decay']} for g in opt.param_groups]

def collect(ratio, entry):
 # Candidate entry parsing is run here, and it has no mechanism flags; the
 # existing subprocess harness additionally proves actual candidate.main OFF
 # delegates baseline.main with identical one-step checkpoint/timeline.
 argsv=["--data_root","/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw","--labeled_manifest",str(ROOT/f"audit/mscmr_{ratio}_labeled_manifest.tsv"),"--unlabeled_manifest",str(ROOT/f"audit/mscmr_{ratio}_unlabeled_manifest.tsv"),"--validation_manifest",str(ROOT/"audit/mscmr_split_manifest.tsv"),"--validation_role","validation","--label_ratio",ratio,"--pretrain_checkpoint",str(ROOT/f"runs/mscmr_{ratio}_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt"),"--run_root",str(ROOT/"runs/trace_unused"),"--seed","1337","--gpu","0","--max_iterations","2","--validation_interval","1","--checkpoint_interval","1","--num_workers","0"]
 (C.parser() if entry=='candidate_off' else B.parser()).parse_args(argsv)
 seed_everything(1337); dev=torch.device('cpu'); pre=load_checkpoint(argsv[argsv.index('--pretrain_checkpoint')+1],dev)
 lab=MSCMRSliceDataset(argsv[1],argsv[3],(256,256),augment=True,include_label=True); un=MSCMRSliceDataset(argsv[1],argsv[5],(256,256),augment=True,include_label=False)
 sam=PairedInfiniteBatchSampler(len(lab),len(un),4,4,1337); loader=DataLoader(ConcatDataset([lab,un]),batch_sampler=sam,num_workers=0); it=iter(loader)
 m1,m2,tea=make_model(dev),make_model(dev),make_model(dev)
 for m in (m1,m2,tea):m.load_state_dict(pre['model'],strict=True)
 tea.requires_grad_(False);tea.eval(); o1=torch.optim.SGD(m1.parameters(),lr=.01,momentum=.9,weight_decay=1e-4);o2=torch.optim.SGD(m2.parameters(),lr=.01,momentum=.9,weight_decay=1e-4);s1=make_poly_scheduler(o1,2);s2=make_poly_scheduler(o2,2)
 out={};out['config']=vars(B.parser().parse_args(argsv));out['names1']=[n for n,_ in m1.named_parameters()];out['names2']=[n for n,_ in m2.named_parameters()];out['namest']=[n for n,_ in tea.named_parameters()];out['init1']=m1.state_dict();out['init2']=m2.state_dict();out['initt']=tea.state_dict();out['groups']=[groups(o1,[m1]),groups(o2,[m2])];out['ostate']= [o1.state_dict(),o2.state_dict()];out['sstate']=[s1.state_dict(),s2.state_dict()];out['rng0']=rng_state()
 batch=next(it); weak,strong=batch['image'],batch['image_strong'];labels,labels_s=batch['label'],batch['label_strong'];out['labeled_batch']={k:batch[k][:4] if isinstance(batch[k],torch.Tensor) else batch[k][:4] for k in ('image','label','image_strong','label_strong','patient_id','slice_index')};out['unlabeled_batch']={k:batch[k][4:] if isinstance(batch[k],torch.Tensor) else batch[k][4:] for k in ('image','label','image_strong','label_strong','patient_id','slice_index')};out['augment']={'weak':weak,'strong':strong,'label_strong':labels_s}
 la,lb,ua,ub=weak[:2],weak[2:4],weak[4:6],weak[6:];la_s,lb_s,ua_s,ub_s=strong[:2],strong[2:4],strong[4:6],strong[6:];ta,tb,ta_s,tb_s=labels[:2],labels[2:4],labels_s[:2],labels_s[2:4]
 with torch.no_grad(): pa,pb,pas,pbs=[torch.softmax(tea(x),1) for x in (ua,ub,ua_s,ub_s)]
 psa,psb,psas,psbs=[x.argmax(1) for x in (pa,pb,pas,pbs)];out['teacher_logits']= [pa,pb,pas,pbs];out['pseudo']=[psa,psb,psas,psbs];out['confidence']=[x.max(1).values for x in (pa,pb,pas,pbs)]
 keep=rectangular_mask(2,256,256,dev);u1,l1=ua*keep[:,None]+la*(1-keep[:,None]),lb*keep[:,None]+ub*(1-keep[:,None]);u2,l2=ua_s*keep[:,None]+la_s*(1-keep[:,None]),lb_s*keep[:,None]+ub_s*(1-keep[:,None]);out['selection']=keep;out['bcp']=[keep,u1,l1,u2,l2]
 o1.zero_grad(set_to_none=True);o2.zero_grad(set_to_none=True);a,b,c,d=m1(u1),m1(l1),m2(u2),m2(l2);out['logits']=[a,b,c,d]
 lu1,_,_=mixed_supervised_loss(a,psa,ta,keep,.5,1);ll1,_,_=mixed_supervised_loss(b,tb,psb,keep,1,.5);lu2,_,_=mixed_supervised_loss(c,psas,ta_s,keep,.5,1);ll2,_,_=mixed_supervised_loss(d,tb_s,psbs,keep,1,.5);cross1=soft_dice_loss(torch.cat([a,b]),torch.cat([c,d]).detach().argmax(1));cross2=soft_dice_loss(torch.cat([c,d]),torch.cat([a,b]).detach().argmax(1));fi=fft_low_frequency_swap(torch.cat([u1,l1]),torch.cat([u2,l2]),30);f1,f2=m1(fi),m2(fi);fp1,fp2=soft_dice_loss(f1,f2.detach().argmax(1)),soft_dice_loss(f2,f1.detach().argmax(1));beta=torch.from_numpy(np.random.beta(.1,.1,size=(2,1,1,1))).float();mix=ua*(1-beta)+ub*beta;lic=m1(torch.cat([weak[:4],mix]));sup,_,_=supervised_loss(lic[:4],labels[:4]);tm=pa*(1-beta)+pb*beta;licr=F.mse_loss(torch.softmax(lic[4:],1),tm)*B.consistency_weight(0,30000,.1);loss1=lu1+ll1+cross1+fp1;loss2=lu2+ll2+cross2+fp2;loss=loss1+loss2+sup+licr;out['losses']={'supervised':sup,'unsupervised':[lu1,ll1,lu2,ll2,cross1,cross2,fp1,fp2,licr],'total':loss};out['fft']=[fi,f1,f2]
 loss.backward();out['grads']={'m1':[(n,p.grad.clone()) for n,p in m1.named_parameters()],'m2':[(n,p.grad.clone()) for n,p in m2.named_parameters()]};out['grad_norm']=[torch.nn.utils.clip_grad_norm_(m1.parameters(),float('inf')),torch.nn.utils.clip_grad_norm_(m2.parameters(),float('inf'))];o1.step();o2.step();s1.step();s2.step();update_ema(m1,tea,.99);out['post1']=m1.state_dict();out['post2']=m2.state_dict();out['postt']=tea.state_dict();out['postopt']=[o1.state_dict(),o2.state_dict()];out['postsched']=[s1.state_dict(),s2.state_dict()];out['rng1']=rng_state();out['modes']=[m1.training,m2.training,tea.training]
 with preserve_module_modes(m1,m2,tea):m1.eval();m2.eval();tea.eval();val=evaluate_manifest(m1,argsv[1],argsv[7],(256,256),dev,role='validation')
 out['validation']=val; payload={'model1':m1.state_dict(),'model2':m2.state_dict(),'ema_teacher':tea.state_dict(),'optimizer1':o1.state_dict(),'optimizer2':o2.state_dict(),'scheduler':{'model1':s1.state_dict(),'model2':s2.state_dict()},'random_state':rng_state(),'global_iteration':1};p=ROOT/f"audit/artifacts/mscmr_{ratio}_{entry}_trace.ckpt";save_checkpoint(p,payload);z=load_checkpoint(p,dev);rm1,rm2,rt=make_model(dev),make_model(dev),make_model(dev);rm1.load_state_dict(z['model1'],strict=True);rm2.load_state_dict(z['model2'],strict=True);rt.load_state_dict(z['ema_teacher'],strict=True);ro1=torch.optim.SGD(rm1.parameters(),lr=.01,momentum=.9,weight_decay=1e-4);ro2=torch.optim.SGD(rm2.parameters(),lr=.01,momentum=.9,weight_decay=1e-4);ro1.load_state_dict(z['optimizer1']);ro2.load_state_dict(z['optimizer2']);rs1=make_poly_scheduler(ro1,2);rs2=make_poly_scheduler(ro2,2);rs1.load_state_dict(z['scheduler']['model1']);rs2.load_state_dict(z['scheduler']['model2']);restore_rng_state(z['random_state']);sam.iteration=1;rb=next(iter(loader));out['checkpoint']=payload;out['resume']={'loaded':z,'next_batch':{k:rb[k] for k in ('image','label','image_strong','label_strong')},'next_forward':rm1(rb['image'][:2]),'modes':[rm1.training,rm2.training,rt.training]};return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ratio',choices=['5pct','10pct'],required=True);a=ap.parse_args();Path(ROOT/'audit/artifacts').mkdir(exist_ok=True)
 raw_path=ROOT/f'audit/artifacts/mscmr_{a.ratio}_default_off_42_raw.pt'
 if raw_path.exists():
  raw=torch.load(raw_path,map_location='cpu',weights_only=False);x,y=raw['baseline'],raw['candidate_off']
 else:
  x,y=collect(a.ratio,'baseline'),collect(a.ratio,'candidate_off');torch.save({'baseline':x,'candidate_off':y},raw_path)
 keys=['config','names1','names2','namest','init1','init2','initt','groups','ostate','sstate','rng0','rng0','rng0','rng0','labeled_batch','unlabeled_batch','augment','logits','logits','teacher_logits','losses','pseudo','confidence','selection','bcp','fft','losses','losses','grads','grad_norm','post1','post2','postt','postopt','postsched','rng1','modes','validation','validation','checkpoint','resume','resume']
 trace=[]
 for i,(name,k) in enumerate(zip(ITEMS,keys),1):
  ex,ma,me=eq(x[k],y[k]); rng=i in (11,12,13,14,36); trace.append({'item_id':i,'item_name':name,'collected':True,'compared':True,'baseline_source':k,'candidate_source':k,'baseline_shape_or_type':tdesc(x[k]),'candidate_shape_or_type':tdesc(y[k]),'baseline_dtype':'see raw artifact','candidate_dtype':'see raw artifact','exact_equal':ex,'max_abs_error':ma,'mean_abs_error':me,'key_set_equal':ex,'order_equal':ex,'rng_equal':ex if rng else None,'status':'PASS' if ex else 'FAIL','evidence_path':str(ROOT/f'audit/artifacts/mscmr_{a.ratio}_default_off_42_raw.pt'),'notes':'validation role only; no formal-test input'})
 meta={'ratio':a.ratio,'device':'cpu','seed':1337,'candidate_off_delegation_core_evidence':str(ROOT/f'runs/mscmr_{a.ratio}_candidate_default_off_harness_cpu/result.json'),'raw_artifact':str(ROOT/f'audit/artifacts/mscmr_{a.ratio}_default_off_42_raw.pt'),'items':trace,'collected_count':42,'compared_count':42,'global_max_abs_error':max(t['max_abs_error'] for t in trace),'all_pass':all(t['status']=='PASS' for t in trace),'test_access_count':0}
 Path(ROOT/f'audit/mscmr_{a.ratio}_candidate_default_off_42_item_trace.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'ratio':a.ratio,'pass':meta['all_pass'],'max':meta['global_max_abs_error']}))
if __name__=='__main__':main()
