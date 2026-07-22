#!/usr/bin/env python3
"""Validation-only representativeness audit; never reads formal_test rows."""
import csv,json,re,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from train_MSCMRSEG.v15_1_mscmr_common import make_model
from train_MSCMRSEG.v15_1_mscmr_dataset import _normalize,read_patient_manifest,resize_slice
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest,predict_case
from train_MSCMRSEG.v15_1_mscmr_protocol import raw_label_to_train_label
R=Path('runs/mscmr_10pct_beta_fft_2d_baseline_bayeseg_fixed_split_seed1337_modefix_rerun'); A=Path('audit'); M=Path('audit/mscmr_split_manifest.tsv'); D=Path('/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw'); dev=torch.device('cuda:0')
spec=[('1000',R/'checkpoints/iter_001000.ckpt'),('1200',R/'checkpoints/best.ckpt'),('late_29000',R/'checkpoints/iter_029000.ckpt'),('final_30000',R/'checkpoints/iter_030000.ckpt')]
def model(path):
 c=torch.load(path,map_location=dev,weights_only=False); m=make_model(dev);m.load_state_dict(c['model1'],strict=True);m.eval();return m
def main():
 vals=[]
 for name,path in spec:
  r=evaluate_manifest(model(path),D,M,(256,256),dev,role='validation');vals.append((name,path,r))
 with (A/'mscmr_run1_representativeness_case_metrics.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['checkpoint','patient_id','mDice','mIoU','HD95','ASD','RV','MYO','LV'])
  for n,p,r in vals:
   for c in r['cases']:w.writerow([n,c['patient_id'],c['mDice'],c['mIoU'],c['HD95'],c['ASD'],c['classes']['RV']['dice'],c['classes']['MYO']['dice'],c['classes']['LV']['dice']])
 peak=dict((c['patient_id'],c['mDice']) for c in vals[1][2]['cases']); late=dict((c['patient_id'],c['mDice']) for c in vals[2][2]['cases']); final=dict((c['patient_id'],c['mDice']) for c in vals[3][2]['cases'])
 with (A/'mscmr_run1_1200_case_contribution.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['patient_id','dice_1200','dice_late','dice_final','contribution_vs_late','contribution_vs_final','leave_one_out_1200'])
  for k in peak:w.writerow([k,peak[k],late[k],final[k],peak[k]-late[k],peak[k]-final[k],np.mean([v for q,v in peak.items() if q!=k])])
 log=[]
 for l in open(R/'console.log'):
  x=re.search(r'VALIDATION iteration=(\d+).*?mDice=([0-9.]+)',l)
  if x:log.append((int(x.group(1)),float(x.group(2))))
 x=[v for _,v in log];latevals=[v for i,v in log if i>=20000]
 stat={'n':len(x),'mean':np.mean(x),'median':np.median(x),'std':np.std(x),'top5_mean':np.mean(sorted(x)[-5:]),'late_stage_definition':'iteration >=20000','late_stage_mean':np.mean(latevals),'late_stage_median':np.median(latevals),'late_stage_std':np.std(latevals),'final':x[-1],'peak':max(log,key=lambda q:q[1]),'late_reference_checkpoint':'iter_029000 (nearest saved checkpoint to late-stage median; its validation event at 29000 is 0.73627641)'}
 (A/'mscmr_run1_validation_trajectory_statistics.json').write_text(json.dumps(stat,indent=2)+'\n')
 rows=read_patient_manifest(M,'validation'); mods=[(n,model(p)) for n,p,_ in vals]
 out=A/'mscmr_run1_1200_late_prediction_visualizations';out.mkdir(exist_ok=True)
 for row in rows:
  im=_normalize(np.asanyarray(nib.load(str(D/row['image_path'])).dataobj));gt=raw_label_to_train_label(np.asanyarray(nib.load(str(D/row['label_path'])).dataobj));z=im.shape[2]//2
  fig,ax=plt.subplots(1,6,figsize=(18,3));ax[0].imshow(im[:,:,z],cmap='gray');ax[0].set_title(row['patient_id']+' image');ax[1].imshow(gt[:,:,z],vmin=0,vmax=3);ax[1].set_title('GT')
  for j,(n,m) in enumerate(mods):ax[j+2].imshow(predict_case(m,im,(256,256),dev)[:,:,z],vmin=0,vmax=3);ax[j+2].set_title(n)
  [a.axis('off') for a in ax];fig.tight_layout();fig.savefig(out/(row['patient_id']+'.png'),dpi=140);plt.close(fig)
if __name__=='__main__':main()
