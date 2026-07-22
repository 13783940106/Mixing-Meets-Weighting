#!/usr/bin/env python3
"""Read-only, validation-only deterministic audit for MSCMR Attempt2 best.ckpt."""
import csv, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_MSCMRSEG.v15_1_mscmr_common import make_model, sha256_file
from train_MSCMRSEG.v15_1_mscmr_dataset import _normalize, read_patient_manifest, resize_slice
from train_MSCMRSEG.v15_1_mscmr_eval import CLASS_NAMES, binary_metrics, evaluate_manifest
from train_MSCMRSEG.v15_1_mscmr_protocol import raw_label_to_train_label

ROOT=Path('runs/mscmr_10pct_beta_fft_2d_baseline_bayeseg_fixed_split_seed1337_modefix_rerun')
AUD=Path('audit'); CKPT=ROOT/'checkpoints/best.ckpt'; MAN=Path('audit/mscmr_split_manifest.tsv')
DATA=Path('/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw'); DEV=torch.device('cuda:0')

def finite(m): return all(torch.isfinite(x).all().item() for x in m.state_dict().values() if torch.is_floating_point(x))
@torch.inference_mode()
def ensemble_case(a,b,image):
    a.eval(); b.eval(); out=np.zeros(image.shape,np.uint8)
    for z in range(image.shape[2]):
        x=torch.from_numpy(resize_slice(image[:,:,z],(256,256),False))[None,None].to(DEV)
        p=((a(x)+b(x))/2).argmax(1)[0].cpu().numpy(); out[:,:,z]=resize_slice(p,image.shape[:2],True)
    return out
def ensemble_eval(a,b):
    cases=[]
    for row in read_patient_manifest(MAN,role='validation'):
        imn=nib.load(str(DATA/row['image_path'])); lan=nib.load(str(DATA/row['label_path']))
        im=_normalize(np.asanyarray(imn.dataobj)); gt=raw_label_to_train_label(np.asanyarray(lan.dataobj)); pr=ensemble_case(a,b,im)
        sp=tuple(float(x) for x in lan.header.get_zooms()[:3]); cl={CLASS_NAMES[c]:binary_metrics(pr==c,gt==c,sp) for c in CLASS_NAMES}
        cases.append({'patient_id':row['patient_id'],'shape':list(gt.shape),'slice_count':int(gt.shape[2]),'classes':cl,'mDice':float(np.mean([x['dice'] for x in cl.values()]))})
    cls={n:{'dice':float(np.mean([x['classes'][n]['dice'] for x in cases]))} for n in CLASS_NAMES.values()}
    return {'case_count':len(cases),'classes':cls,'cases':cases,'mDice':float(np.mean([cls[n]['dice'] for n in cls]))}
def main():
    ck=torch.load(CKPT,map_location=DEV,weights_only=False); models={}
    for name,key in [('model1','model1'),('model2','model2'),('ema_teacher','ema_teacher')]:
        m=make_model(DEV); m.load_state_dict(ck[key],strict=True); m.eval(); models[name]=m
    rows=[]; details=[]
    for repeat in range(1,4):
        r=evaluate_manifest(models['model1'],DATA,MAN,(256,256),DEV,role='validation'); rows.append({'replay_id':repeat,'source':'model1_formal_selected','mDice':r['mDice']}); details.append((repeat,'model1_formal_selected',r))
    for name in ('model2','ema_teacher'):
        r=evaluate_manifest(models[name],DATA,MAN,(256,256),DEV,role='validation'); rows.append({'replay_id':1,'source':name,'mDice':r['mDice']}); details.append((1,name,r))
    r=ensemble_eval(models['model1'],models['model2']); rows.append({'replay_id':1,'source':'diagnostic_logit_mean_ensemble','mDice':r['mDice']}); details.append((1,'diagnostic_logit_mean_ensemble',r))
    with (AUD/'mscmr_run1_1200_replay_runs.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['replay_id','source','mDice'],delimiter='\t');w.writeheader();w.writerows(rows)
    with (AUD/'mscmr_run1_1200_case_metrics.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['replay_id','source','patient_id','slice_count','mDice','RV_dice','MYO_dice','LV_dice'])
        for rep,src,x in details:
            for c in x['cases']: w.writerow([rep,src,c['patient_id'],c.get('slice_count',c['shape'][2]),c['mDice'],c['classes']['RV']['dice'],c['classes']['MYO']['dice'],c['classes']['LV']['dice']])
    with (AUD/'mscmr_run1_1200_model_source_comparison.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['replay_id','source','mDice'],delimiter='\t');w.writeheader();w.writerows(rows)
    formal=[x['mDice'] for x in rows if x['source']=='model1_formal_selected']; ok=max(formal)-min(formal)<=1e-6 and abs(formal[0]-0.82693768)<=1e-6
    report={'checkpoint':str(CKPT),'sha256':sha256_file(CKPT),'global_iteration':ck['global_iteration'],'best_iteration':ck['best_iteration'],'best_metric':ck['best_validation_metric'],'best_model_name':ck['best_model_name'],'split_sha256':sha256_file(MAN),'validation_patients':[r['patient_id'] for r in read_patient_manifest(MAN,'validation')],'models_finite':{k:finite(v) for k,v in models.items()},'formal_replays':formal,'reproduced':ok,'source_rows':rows}
    (AUD/'mscmr_1200_replay_raw.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
