import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_MSCMRSEG.v15_1_mscmr_common import make_model,sha256_file
from train_MSCMRSEG.v15_1_mscmr_candidate_modules import MaskTokenAdapter,DecoderHeadRouter
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest
R=Path('runs/mscmr_5pct_beta_fft_mask_token_adamix_bayeseg_fixed_split_seed1337');C=R/'checkpoints/best.ckpt';M=Path('audit/mscmr_split_manifest.tsv');D='/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw';A=Path('audit');dev=torch.device('cuda:0')
rows=[];cases=[]
for n in range(1,4):
 x=torch.load(C,map_location=dev,weights_only=False);assert x['global_iteration']==800 and x['best_model_name']=='model1'
 m=make_model(dev);m.load_state_dict(x['model1'],strict=True); token=MaskTokenAdapter(16,x['candidate_metadata']['innovation_config']['mask_token_num']).to(dev);token.load_state_dict(x['mask_token_model1'],strict=True);hook=DecoderHeadRouter(m,token);hook.attach();m.eval()
 assert all(torch.isfinite(v).all() for v in m.state_dict().values() if torch.is_floating_point(v));assert all(torch.isfinite(v).all() for v in token.state_dict().values())
 r=evaluate_manifest(m,D,M,(256,256),dev,role='validation');hook.remove();rows.append([n,r['mDice'],r['classes']['RV']['dice'],r['classes']['MYO']['dice'],r['classes']['LV']['dice'],r['case_count']])
 for c in r['cases']:cases.append([n,c['patient_id'],c['mDice'],c['classes']['RV']['dice'],c['classes']['MYO']['dice'],c['classes']['LV']['dice'],c['shape'][2]])
with (A/'mscmr_5pct_candidate_replay_runs.tsv').open('w',newline='') as f:w=csv.writer(f,delimiter='\t');w.writerow(['replay','mDice','RV','MYO','LV','case_count']);w.writerows(rows)
with (A/'mscmr_5pct_candidate_replay_cases.tsv').open('w',newline='') as f:w=csv.writer(f,delimiter='\t');w.writerow(['replay','patient','mDice','RV','MYO','LV','slices']);w.writerows(cases)
errs=[max(r[i] for r in rows)-min(r[i] for r in rows) for i in range(1,5)];out={'global_max_abs_error':max(errs),'rows':rows,'checkpoint_sha256':sha256_file(C),'evaluator_sha256':sha256_file('train_MSCMRSEG/v15_1_mscmr_eval.py'),'manifest_sha256':sha256_file(M),'pass':max(errs)<=1e-6 and all(abs(r[1]-.8156211943010504)<=1e-6 for r in rows),'test_access_count':0}
(A/'mscmr_5pct_candidate_replay.json').write_text(json.dumps(out,indent=2)+'\n')
