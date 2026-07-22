import json,sys,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from train_MSCMRSEG.v15_1_mscmr_common import make_model,sha256_file
from train_MSCMRSEG.v15_1_mscmr_candidate_modules import MaskTokenAdapter,DecoderHeadRouter
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

R=Path('runs/mscmr_5pct_beta_fft_mask_token_adamix_bayeseg_fixed_split_seed1337')
C=R/'checkpoints/best.ckpt';M=Path('audit/mscmr_split_manifest.tsv')
D='/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw';A=Path('audit')
OUT=A/'mscmr_5pct_candidate_one_time_formal_test.json';P=A/'mscmr_5pct_candidate_formal_test_predictions'
if OUT.exists(): raise RuntimeError('formal test already consumed')
x=torch.load(C,map_location='cuda:0',weights_only=False)
assert x['global_iteration']==800 and x['best_model_name']=='model1'
m=make_model('cuda:0');m.load_state_dict(x['model1'],strict=True)
tok=MaskTokenAdapter(16,4).to('cuda:0');tok.load_state_dict(x['mask_token_model1'],strict=True)
hook=DecoderHeadRouter(m,tok);hook.attach();m.eval()
r=evaluate_manifest(m,D,M,(256,256),'cuda:0',include_predictions=True,role='formal_test');hook.remove();P.mkdir(exist_ok=True)
for case in r['cases']:
    np.save(P/f"{case['patient_id']}_prediction.npy",case.pop('prediction'))
r.update({'checkpoint':str(C),'checkpoint_sha256':sha256_file(C),'global_iteration':800,'prediction_source':'model1','candidate_trainer_sha256':sha256_file('train_MSCMRSEG/v15_1_mscmr_mask_token_adamix_candidate.py'),'candidate_modules_sha256':sha256_file('train_MSCMRSEG/v15_1_mscmr_candidate_modules.py'),'evaluator_sha256':sha256_file('train_MSCMRSEG/v15_1_mscmr_eval.py'),'pretrain_sha256':sha256_file('runs/mscmr_5pct_pretrain_bayeseg_fixed_split_seed1337/checkpoints/best.ckpt'),'test_manifest_sha256':sha256_file(M),'formal_test_consumed':True,'timestamp':datetime.datetime.now().isoformat(),'prediction_directory':str(P)})
OUT.write_text(json.dumps(r,indent=2)+'\n')
