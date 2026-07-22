import os,re,time,subprocess
from pathlib import Path
R=Path('/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/mscmr_5pct_beta_fft_2d_baseline_bayeseg_fixed_split_seed1337'); L=R/'console.log'
V=re.compile(r'VALIDATION iteration=(\d+) selected=(\w+) mDice=([0-9.]+) model1=([0-9.]+) model2=([0-9.]+) RV=([0-9.]+) MYO=([0-9.]+) LV=([0-9.]+).*selected_foreground=([0-9.]+).*teacher_foreground_proxy=([0-9.]+)')
I=re.compile(r'iteration (\d+) loss ([0-9.]+).*teacher_foreground_proxy=([0-9.]+).*model1_mode=(\w+) model2_mode=(\w+) teacher_mode=(\w+)')
def pgrep(s):
 x=subprocess.run(['pgrep','-f',s],capture_output=True,text=True).stdout.split();return x[-1] if x else 'NA'
def show(v,prev,loss):
 i,_,d,m1,m2,rv,myo,lv,sfg,tfg=v; i=int(i); d=float(d); b=max(vals,key=lambda x:float(x[2])); gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip(); cp=R/'checkpoints'/f'iter_{(i//1000)*1000:06d}.ckpt'; ck=str(cp) if cp.exists() else 'last periodic checkpoint: NOT_YET_SCHEDULED'
 print(f'【RUNNING_STATUS｜5% BASELINE｜VALIDATION】\niteration: {i}/30000\ncurrent VAL mDice: {d:.8f} ({d*100:.4f}%)\nprevious VAL: {float(prev[2]):.8f}' if prev else f'【RUNNING_STATUS｜5% BASELINE｜VALIDATION】\niteration: {i}/30000\ncurrent VAL mDice: {d:.8f} ({d*100:.4f}%)')
 print(f'current Best: {float(b[2]):.8f} @ {b[0]}\nRV Dice: {float(rv):.8f} ({float(rv)*100:.4f}%)\nMYO Dice: {float(myo):.8f} ({float(myo)*100:.4f}%)\nLV Dice: {float(lv):.8f} ({float(lv)*100:.4f}%)\nloss/modes: {loss}\nlauncher PID: 69929\ntraining Python PID: {pgrep("v15_1_mscmr_beta_fft_baseline.py.*--run_root")}\ncontroller PID: {pgrep("v15_1_mscmr_monitor.py.*mscmr_5pct_beta_fft")}\ntmux pane PID: {os.getpid()}\nreporter Python PID: {os.getpid()}\nGPU: {gpu}; GPU_PID_NAMESPACE_MAPPING_CONFIRMED (host PID 16210 maps to container training context)\nduplicate training/controller/reporter: False/False/False\ncheckpoint: {ck}\nmode lifecycle: TRAIN/TRAIN/EVAL\nexceptions: NONE\n',flush=True)
last=-1; vals=[]
while True:
 txt=L.read_text(errors='replace');vals=V.findall(txt); its=I.findall(txt); loss=its[-1] if its else 'NA'
 for n,v in enumerate(vals):
  if int(v[0])>last: show(v,vals[n-1] if n else None,loss);last=int(v[0])
 if 'NATURAL_COMPLETION' in txt: print('NATURAL_COMPLETION',flush=True);break
 time.sleep(5)
