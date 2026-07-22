#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from train_MSCMRSEG.v15_1_mscmr_common import load_checkpoint, make_model, sha256_file
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run_root',required=True); p.add_argument('--data_root',required=True); p.add_argument('--validation_manifest',required=True); p.add_argument('--output',required=True); p.add_argument('--gpu',default='0'); a=p.parse_args()
 root=Path(a.run_root); best_path=root/'checkpoints/best.ckpt'; last_path=root/'checkpoints/last.ckpt'
 best=load_checkpoint(best_path,'cpu'); last=load_checkpoint(last_path,'cpu'); device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
 model=make_model(device); model.load_state_dict(best['model'],strict=True)
 replay=evaluate_manifest(model,a.data_root,a.validation_manifest,(256,256),device,role='validation')
 source_match={path: (Path(path).exists() and sha256_file(path)==digest) for path,digest in best['source_sha256'].items()}
 natural='NATURAL_COMPLETION iteration=10000' in (root/'console.log').read_text()
 decision=natural and last['global_iteration']==10000 and best['best_iteration']==4600 and abs(replay['mDice']-best['best_validation_metric'])<1e-8 and all(source_match.values())
 result={'decision':'PASS' if decision else 'FAIL','natural_completion':natural,'best_checkpoint':str(best_path),'best_sha256':sha256_file(best_path),'last_checkpoint':str(last_path),'last_sha256':sha256_file(last_path),'best_validation_metric':best['best_validation_metric'],'best_iteration':best['best_iteration'],'final_validation':last['last_validation'],'strict_load':True,'validation_replay':replay,'validation_replay_delta':replay['mDice']-best['best_validation_metric'],'source_sha_match':source_match,'split_sha256':best['split_sha256'],'pretrain_format':best['format']}
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=True)+'\n'); print(json.dumps({'decision':result['decision'],'best':result['best_validation_metric'],'iteration':result['best_iteration'],'replay':replay['mDice'],'sha256':result['best_sha256']})); return 0 if decision else 2
if __name__=='__main__': raise SystemExit(main())
