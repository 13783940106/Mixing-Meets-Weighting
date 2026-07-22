#!/usr/bin/env python3
"""Read-only verifier for a completed validation-only ON engineering smoke."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from train_MSCMRSEG.v15_1_mscmr_candidate_modules import MaskTokenAdapter
from train_MSCMRSEG.v15_1_mscmr_common import load_checkpoint

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    root=Path(a.run_root); ckpt=load_checkpoint(root/'checkpoints/last.ckpt','cpu'); timeline=json.loads((root/'training_timeline.jsonl').read_text().splitlines()[0])
    cfg=ckpt['candidate_metadata']['innovation_config']; assert cfg['enable_mask_token'] and cfg['enable_adamix']
    for key in ('mask_token_model1','mask_token_model2'):
        module=MaskTokenAdapter(16,cfg['mask_token_num']); module.load_state_dict(ckpt[key],strict=True)
    mt=timeline['mask_token']; am=timeline['adamix']; passed=(mt['grad_norm']>0 and mt['update_max_abs_delta']>0 and am['trigger_count']>0 and am['entered_unsupervised_loss'] and timeline['fft_finite'] and timeline['ema_finite'])
    out={'run_root':str(root),'formal_test_access':False,'validation_only':True,'strict_load':True,'resume_checkpoint_available':(root/'checkpoints/last.ckpt').exists(),'mask_token':mt,'adamix':am,'finite':bool(timeline['fft_finite'] and timeline['ema_finite']),'pass':bool(passed)}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); return 0 if passed else 1
if __name__=='__main__': raise SystemExit(main())
