#!/usr/bin/env python3
"""Validation-only checkpoint evolution audit after SELF_TRAINING_DESTROYS_PRETRAIN."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from train_MSCMRSEG.v15_1_mscmr_common import load_checkpoint,make_model,sha256_file
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

def mean_fg(metrics):return float(np.mean([x['predicted_foreground_ratio'] for x in metrics['cases']]))
def drift(state,reference):
    total=ref=0.
    for key,value in state.items():
        if torch.is_floating_point(value):total+=float((value-reference[key]).double().square().sum());ref+=float(reference[key].double().square().sum())
    return math.sqrt(total),math.sqrt(total/max(ref,1e-30))
def compact(metrics):return {'mDice':metrics['mDice'],'classes':{k:v['dice'] for k,v in metrics['classes'].items()},'mean_predicted_foreground_ratio':mean_fg(metrics)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--pretrain',required=True);p.add_argument('--data-root',required=True);p.add_argument('--validation-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');a=p.parse_args();root=Path(a.run_root);device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu');pre=load_checkpoint(a.pretrain,'cpu')['model'];rows=[]
 paths=[root/'checkpoints'/f'iter_{i:06d}.ckpt' for i in range(1000,6001,1000)]
 for path in paths:
  c=load_checkpoint(path,'cpu');teacher=make_model(device);teacher.load_state_dict(c['ema_teacher'],strict=True);teacher_metrics=evaluate_manifest(teacher,a.data_root,a.validation_manifest,(256,256),device,role='validation');del teacher;torch.cuda.empty_cache()
  item={'iteration':c['global_iteration'],'sha256':sha256_file(path),'selected_model':c['last_validation']['selected_model'],'selected':compact(c['last_validation']['selected']),'model1':compact(c['last_validation']['model1']),'model2':compact(c['last_validation']['model2']),'ema_teacher':compact(teacher_metrics),'drift_from_pretrain':{}}
  for name in ('model1','model2','ema_teacher'):
   absolute,relative=drift(c[name],pre);item['drift_from_pretrain'][name]={'l2':absolute,'relative_l2':relative}
  item['lr']={'model1':c['optimizer1']['param_groups'][0]['lr'],'model2':c['optimizer2']['param_groups'][0]['lr']};item['mechanisms']=c['mechanisms'];rows.append(item)
 result={'decision':'SELF_TRAINING_DESTROYS_PRETRAIN_CONFIRMED','selection_data':'validation_only','formal_test_accessed':False,'pretrain_mDice':0.6232621473953949,'checkpoints':rows}
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=True)+'\n');print(json.dumps({'decision':result['decision'],'evolution':[(x['iteration'],x['model1']['mDice'],x['model2']['mDice'],x['ema_teacher']['mDice']) for x in rows]}));return 0
if __name__=='__main__':raise SystemExit(main())
