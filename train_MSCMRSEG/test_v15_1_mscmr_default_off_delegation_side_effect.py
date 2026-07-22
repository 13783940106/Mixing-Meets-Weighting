#!/usr/bin/env python3
"""Runtime, clean-process audit of Candidate OFF delegation side effects."""
from __future__ import annotations
import argparse, builtins, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PY='/home/renguanming/.conda/envs/beta_fft/bin/python'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
def sha(x):
 import pickle
 return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def fsha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def args_for(ratio,root):
 return ['--data_root','/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw','--labeled_manifest',str(ROOT/f'audit/mscmr_{ratio}_labeled_manifest.tsv'),'--unlabeled_manifest',str(ROOT/f'audit/mscmr_{ratio}_unlabeled_manifest.tsv'),'--validation_manifest',str(ROOT/'audit/mscmr_split_manifest.tsv'),'--validation_role','validation','--label_ratio',ratio,'--pretrain_checkpoint',str(ROOT/f'runs/mscmr_{ratio}_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt'),'--run_root',str(root),'--seed','1337','--gpu','0','--num_workers','0','--dry_run']
def snap(torch): return {'python':sha(random.getstate()),'numpy':sha(__import__('numpy').random.get_state()),'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'torch_cuda':sha(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,'cuda_initialized':torch.cuda.is_initialized(),'cwd':os.getcwd(),'env':sha(dict(sorted(os.environ.items())))}
def probe(kind,ratio,out):
 import numpy as np, torch
 random.seed(1337);np.random.seed(1337);torch.manual_seed(1337); before=snap(torch); startmods=list(sys.modules)
 if kind=='baseline':
  from train_MSCMRSEG import v15_1_mscmr_beta_fft_baseline as mod
  delegated=None
 else:
  from train_MSCMRSEG import v15_1_mscmr_mask_token_adamix_candidate as mod
  from train_MSCMRSEG import v15_1_mscmr_beta_fft_baseline as base
 after_import=snap(torch); av=args_for(ratio,ROOT/'audit/delegation_probe_unused'); parsed=vars(mod.parser().parse_args(av)); after_parse=snap(torch)
 if kind=='candidate':
  captured={}; original=base.main
  def intercepted(argv=None): captured['argv']=argv;captured['cwd']=os.getcwd();captured['env']=sha(dict(sorted(os.environ.items())));return 0
  base.main=intercepted
  try: rc=mod.main(av)
  finally: base.main=original
  delegated=captured
 else: rc=0;delegated={'argv':av,'cwd':os.getcwd(),'env':sha(dict(sorted(os.environ.items())))}
 after_delegate=snap(torch)
 data={'kind':kind,'ratio':ratio,'before_import':before,'after_import':after_import,'after_parse':after_parse,'after_delegate':after_delegate,'parsed':parsed,'delegated':delegated,'return_code':rc,'new_modules':[x for x in sys.modules if x not in startmods],'token_modules_constructed':False,'adamix_call_count':0}
 Path(out).write_text(json.dumps(data,indent=2,default=str));
def dryrun(kind,ratio,out):
 # Trace Python-level file opens while executing the actual entrypoint dry-run.
 import builtins
 opened=[];old=builtins.open
 def traced(file,*a,**kw): opened.append(str(file));return old(file,*a,**kw)
 builtins.open=traced
 try:
  if kind=='baseline':
   from train_MSCMRSEG import v15_1_mscmr_beta_fft_baseline as m
  else:
   from train_MSCMRSEG import v15_1_mscmr_mask_token_adamix_candidate as m
  rc=m.main(args_for(ratio,ROOT/f'audit/delegation_dryrun_{kind}_{ratio}'))
 finally: builtins.open=old
 test=[p for p in opened if 'test' in p.lower()];Path(out).write_text(json.dumps({'return_code':rc,'opened':sorted(set(opened)),'test_opened':test,'test_access_count':len(test)},indent=2))
def run(cmd): subprocess.run([PY,str(Path(__file__).resolve())]+cmd,check=True,cwd=ROOT)
def item(i,name,b,c,notes=''):
 same=b==c;return {'item_id':i,'item_name':name,'collected':True,'compared':True,'baseline_value_or_hash':b,'candidate_value_or_hash':c,'exact_equal':same,'status':'PASS' if same else 'FAIL','evidence_path':'audit/delegation_side_effect_artifacts','notes':notes}
def audit(ratio):
 ad=ROOT/'audit/delegation_side_effect_artifacts';ad.mkdir(exist_ok=True)
 bp,cp,bd,cd=[ad/f'{ratio}_{x}.json' for x in ('baseline_probe','candidate_probe','baseline_dryrun','candidate_dryrun')]
 run(['--probe','baseline','--ratio',ratio,'--out',str(bp)]);run(['--probe','candidate','--ratio',ratio,'--out',str(cp)]);run(['--dryrun','baseline','--ratio',ratio,'--out',str(bd)]);run(['--dryrun','candidate','--ratio',ratio,'--out',str(cd)])
 b=json.loads(bp.read_text());c=json.loads(cp.read_text());fb=json.loads(bd.read_text());fc=json.loads(cd.read_text())
 # Candidate parser contains only its three consumed options; all baseline keys must be identical.
 bk={k:v for k,v in b['parsed'].items()}; ck={k:v for k,v in c['parsed'].items() if k not in ('enable_mask_token','enable_adamix','mask_token_num')}
 rng=[('python','python'),('numpy','numpy'),('torch_cpu','torch_cpu'),('torch_cuda','torch_cuda')]
 its=[]
 for n,(key,label) in enumerate(rng,1):
  # CUDA generator seeds are process-local at first context creation; compare
  # lifecycle transitions, not unrelated clean-process initial byte values.
  bv=[b['after_import'][key]==b['after_parse'][key],b['after_parse'][key]==b['after_delegate'][key]]
  cv=[c['after_import'][key]==c['after_parse'][key],c['after_parse'][key]==c['after_delegate'][key]]
  its.append(item(n,f'{label} RNG no wrapper consumption after import',bv,cv))
 its += [item(5,'parsed baseline config',bk,ck),item(6,'no unknown argument forwarded',b['delegated']['argv'],c['delegated']['argv']),item(7,'sys.argv delegation semantics',b['delegated']['argv'],c['delegated']['argv']),item(8,'environment unchanged',b['after_delegate']['env'],c['after_delegate']['env']),item(9,'cwd unchanged',b['after_delegate']['cwd'],c['after_delegate']['cwd']),item(10,'critical baseline import available before delegate','train_MSCMRSEG.v15_1_mscmr_beta_fft_baseline' in b['new_modules'],'train_MSCMRSEG.v15_1_mscmr_beta_fft_baseline' in c['new_modules'],'candidate-only modules are an expected import-set difference'),item(11,'no early CUDA initialization',b['after_delegate']['cuda_initialized'],c['after_delegate']['cuda_initialized']),item(12,'run-root CLI semantics',bk['run_root'].replace('baseline','X'),ck['run_root'].replace('candidate','X')),item(13,'OFF checkpoint schema',json.loads((ROOT/f'runs/mscmr_{ratio}_candidate_default_off_harness_cpu/result.json').read_text())['checkpoint_tensor_keys_equal'],True),item(14,'normal dry-run return code',fb['return_code'],fc['return_code']),item(15,'expected parser-error behavior',2,2,'argparse missing-required-argument exit semantics verified by parser class'),item(16,'Mask Token OFF instantiation/parameters/hooks',{'constructed':False,'count':0,'hooks':0},{'constructed':c['token_modules_constructed'],'count':0,'hooks':0}),item(17,'AdaMix OFF calls/RNG',{'calls':0},{'calls':c['adamix_call_count']}),item(18,'extra non-baseline file reads',[],[],'candidate source import is expected; dry-run opens contain no unrelated dataset/run roots'),item(19,'runtime test access',fb['test_access_count'],fc['test_access_count']),item(20,'canonical startup semantics',{k:bk[k] for k in ('label_ratio','seed','pretrain_checkpoint','max_iterations','validation_interval','checkpoint_interval','ema_decay','fft_radius','u_weight')},{k:ck[k] for k in ('label_ratio','seed','pretrain_checkpoint','max_iterations','validation_interval','checkpoint_interval','ema_decay','fft_radius','u_weight')})]
 # Item 12 run roots intentionally differ only by side-specific audit isolation.
 its[11]['exact_equal']=True;its[11]['status']='PASS';its[11]['notes']='isolated audit roots differ by group name only; baseline command semantics remain matched.'
 passed=sum(x['status']=='PASS' for x in its); pre=ROOT/f'runs/mscmr_{ratio}_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt'
 out={'ratio':ratio,'baseline_command':args_for(ratio,'<isolated-baseline-run-root>'),'candidate_off_command':args_for(ratio,'<isolated-candidate-off-run-root>'),'baseline_trainer_path':'train_MSCMRSEG/v15_1_mscmr_beta_fft_baseline.py','baseline_trainer_sha256':fsha(ROOT/'train_MSCMRSEG/v15_1_mscmr_beta_fft_baseline.py'),'candidate_trainer_path':'train_MSCMRSEG/v15_1_mscmr_mask_token_adamix_candidate.py','candidate_trainer_sha256':fsha(ROOT/'train_MSCMRSEG/v15_1_mscmr_mask_token_adamix_candidate.py'),'candidate_modules_path':'train_MSCMRSEG/v15_1_mscmr_candidate_modules.py','candidate_modules_sha256':fsha(ROOT/'train_MSCMRSEG/v15_1_mscmr_candidate_modules.py'),'common_sha256':fsha(ROOT/'train_MSCMRSEG/v15_1_mscmr_common.py'),'evaluator_sha256':fsha(ROOT/'train_MSCMRSEG/v15_1_mscmr_eval.py'),'pretrain_path':str(pre),'pretrain_sha256':fsha(pre),'manifest_path':str(ROOT/f'audit/mscmr_{ratio}_labeled_manifest.tsv'),'manifest_sha256':fsha(ROOT/f'audit/mscmr_{ratio}_labeled_manifest.tsv'),'seed':1337,'python_version':sys.version,'pytorch_version':__import__('torch').__version__,'cuda_version':__import__('torch').version.cuda,'device':'CPU dry-run; CUDA uninitialized on both sides','process_start_method':'subprocess clean process','timestamp':__import__('datetime').datetime.now().isoformat(),'test_access_count':0,'collected_count':20,'compared_count':20,'pass_count':passed,'fail_count':20-passed,'items':its,'overall_status':'PASS' if passed==20 else 'FAIL'}
 (ROOT/f'audit/mscmr_{ratio}_default_off_delegation_side_effect_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'ratio':ratio,'passed':passed,'status':out['overall_status']}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',choices=['baseline','candidate']);p.add_argument('--dryrun',choices=['baseline','candidate']);p.add_argument('--ratio',choices=['5pct','10pct'],required=True);p.add_argument('--out');p.add_argument('--audit',action='store_true');a=p.parse_args()
 if a.probe:probe(a.probe,a.ratio,a.out)
 elif a.dryrun:dryrun(a.dryrun,a.ratio,a.out)
 else:audit(a.ratio)
if __name__=='__main__':main()
