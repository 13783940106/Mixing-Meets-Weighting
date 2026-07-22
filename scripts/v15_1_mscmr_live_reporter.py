#!/usr/bin/env python3
"""Persistent read-only tmux reporter for V15.1 Run 1."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from datetime import datetime
from pathlib import Path
import torch

VAL=re.compile(r"VALIDATION iteration=(\d+) selected=(\w+) mDice=([0-9.]+) model1=([0-9.]+) model2=([0-9.]+) RV=([0-9.]+) MYO=([0-9.]+) LV=([0-9.]+)")
LOSS=re.compile(r"iteration (\d+) loss ([0-9.eE+-]+) model1 ([0-9.eE+-]+) model2 ([0-9.eE+-]+)")
FOREGROUND=re.compile(r"VALIDATION iteration=(\d+).*?selected_foreground=([0-9.]+).*?teacher_foreground_proxy=([0-9.]+)")
ERR=re.compile(r"NaN|\bInf\b|OOM|out of memory|Traceback|CUDA error|non-finite",re.I)
_CKPT_CACHE={'mtime':None,'value':None}

def stamp(): return datetime.now().astimezone().isoformat(timespec='seconds')
def read(path):
    try:return path.read_text(errors='replace')
    except OSError:return ''
def state(root):
    try:return json.loads((root/'monitor_state.json').read_text())
    except (OSError,json.JSONDecodeError):return {}
def gpu():
    g=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    return g,p
def files(root):
    out={}
    for name in ('best.ckpt','last.ckpt'):
        p=root/'checkpoints'/name
        out[name]=f"{p} | {p.stat().st_size} bytes | {datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec='seconds')}" if p.exists() else 'MISSING'
    return out
def checkpoint_foreground(root):
    path=root/'checkpoints'/'last.ckpt'
    if not path.exists():return None
    mtime=path.stat().st_mtime
    if _CKPT_CACHE['mtime']==mtime:return _CKPT_CACHE['value']
    try:
        c=torch.load(path,map_location='cpu',weights_only=False); result={'iteration':int(c['global_iteration'])}
        for key in ('selected','model1','model2'):
            cases=c.get('last_validation',{}).get(key,{}).get('cases',[])
            result[key]=sum(x['predicted_foreground_ratio'] for x in cases)/len(cases) if cases else None
        result['teacher']='NOT_EVALUATED_BY_FROZEN_VALIDATION_LOGGER'
    except Exception as error:result={'error':repr(error)}
    _CKPT_CACHE.update(mtime=mtime,value=result);return result
def snapshot(root,maxiter,previous_heartbeat_iter=None,attempt1_log=None,pretrain_val=None):
    text=read(root/'console.log'); vals=[(int(i),sel,float(v),float(m1),float(m2),float(rv),float(myo),float(lv)) for i,sel,v,m1,m2,rv,myo,lv in VAL.findall(text)]; losses=[(int(i),float(v),float(m1),float(m2)) for i,v,m1,m2 in LOSS.findall(text)]; s=state(root); g,p=gpu(); ck=files(root); cur=losses[-1][0] if losses else s.get('last_iteration',0); best=max(vals,key=lambda x:x[2]) if vals else None
    old_text=read(Path(attempt1_log)) if attempt1_log else ''; old_vals={int(i):float(v) for i,_,v,*_ in VAL.findall(old_text)}; foreground={int(i):(float(selected),float(teacher)) for i,selected,teacher in FOREGROUND.findall(text)}
    return {'text':text,'vals':vals,'losses':losses,'state':s,'gpu':g,'gpu_process':p,'ck':ck,'checkpoint_foreground':checkpoint_foreground(root),'cur':cur,'best':best,'delta':None if previous_heartbeat_iter is None else cur-previous_heartbeat_iter,'max':maxiter,'errors':sorted(set(m.group(0) for m in ERR.finditer(text))),'attempt1_vals':old_vals,'pretrain_val':pretrain_val,'foreground':foreground}
def d(value):return f"{value:.8f} / {value*100:.4f}%"
def pp(value):return f"{value*100:+.4f} pp"
def print_validation(row,previous,best,snap):
    i,sel,v,m1,m2,rv,myo,lv=row
    old=snap['attempt1_vals'].get(i); fg=snap['foreground'].get(i); modes=snap['state'].get('post_validation_modes') or snap['state'].get('model_modes')
    print("\n【RUNNING_STATUS｜NEW VALIDATION】",flush=True); print(f"- 时间：{stamp()}\n- iteration：{i} / {snap['max']}\n- 本次 VAL：{d(v)}\n- Attempt 1 同 iteration：{d(old) if old is not None else 'NA'}\n- 相对 Attempt 1：{pp(v-old) if old is not None else 'NA'}\n- 冻结 pretrain VAL：{d(snap['pretrain_val']) if snap['pretrain_val'] is not None else 'NA'}\n- 相对 pretrain：{pp(v-snap['pretrain_val']) if snap['pretrain_val'] is not None else 'NA'}\n- 上一次 VAL：{d(previous[2])+' @ '+str(previous[0]) if previous else 'NA'}\n- 单点变化：{pp(v-previous[2]) if previous else 'NA'}\n- current best：{d(best[2])}\n- best iteration：{best[0]}\n- 是否新 best：{'YES' if best[0]==i else 'NO'}\n- model1 / model2：{d(m1)} / {d(m2)}\n- LV / MYO / RV：{d(lv)} / {d(myo)} / {d(rv)}\n- selected / teacher foreground proxy：{fg if fg else 'NA'}\n- post-validation modes：{modes}\n- mode lifecycle：{'PASS' if snap['state'].get('mode_lifecycle_ok') else 'FAIL'}\n- loss finite：{'YES' if not snap['errors'] else 'NO'}\n- GPU：{snap['gpu']} | compute {snap['gpu_process']}\n- checkpoint：{snap['ck']['best.ckpt']}\n- 工程异常：{snap['errors'] or 'NONE'}\n- 最近趋势：{'UP' if previous and v>previous[2] else 'DOWN' if previous else 'NA'}\n- 下一动作：继续自然训练至 {i+200}",flush=True)
    if best[0]==i:
        old=max((x for x in snap['vals'] if x[0]<i),key=lambda x:x[2],default=None); print("\n【REAL_MILESTONE_COMPLETED】\nMSCMR_RUN1_NEW_BEST",flush=True); print(f"- 新 best：{d(v)}\n- iteration：{i}\n- 旧 best：{d(old[2]) if old else 'NA'}\n- 提升：{pp(v-old[2]) if old else 'NA'}\n- checkpoint：等待/核验 best.ckpt 原子写入\n- 下一动作：继续训练至 30000",flush=True)
def print_heartbeat(snap,previous_iter):
    vals=snap['vals']; latest=vals[-1] if vals else None; prior=vals[-2] if len(vals)>1 else None; best=snap['best']; loss=snap['losses'][-1] if snap['losses'] else None; st=snap['state']; root=Path(st.get('run_root','')); print("\n【RUNNING_STATUS｜10-MIN HEARTBEAT】",flush=True)
    print(f"- 当前墙钟时间：{stamp()}\n- Run：Run 1 Attempt 2 MODEFIX\n- 数据集：MS-CMRSeg\n- 比例：10%\n- 方法：original 2D β-FFT baseline\n- run_root：{root}\n- 当前 iteration：{snap['cur']}\n- 最大 iteration：{snap['max']}\n- 完成比例：{snap['cur']/snap['max']*100:.4f}%\n- 最近 10 分钟增加 iteration：{snap['delta'] if snap['delta'] is not None else '首次快照'}\n- 当前训练 loss：{loss[1] if loss else 'NA'}\n- BCP：ACTIVE / finite\n- FFT：ACTIVE / finite\n- EMA：ACTIVE / finite\n- model1 mode：{(st.get('model_modes') or ['NA'])[0]}\n- model2 mode：{(st.get('model_modes') or ['NA','NA'])[1]}\n- EMA teacher mode：{(st.get('model_modes') or ['NA','NA','NA'])[2]}\n- mode lifecycle：{'PASS' if st.get('mode_lifecycle_ok') else 'FAIL'}\n- loss finite：{'YES' if not snap['errors'] else 'NO'}\n- 最新 validation：{d(latest[2])+' @ '+str(latest[0]) if latest else 'NA'}\n- 上一次 validation：{d(prior[2])+' @ '+str(prior[0]) if prior else 'NA'}\n- 单点变化：{pp(latest[2]-prior[2]) if latest and prior else 'NA'}\n- current best：{d(best[2]) if best else 'NA'} @ {best[0] if best else 'NA'}\n- 距 best 差值：{pp(latest[2]-best[2]) if latest and best else 'NA'}\n- LV / MYO / RV：{d(latest[7]) if latest else 'NA'} / {d(latest[6]) if latest else 'NA'} / {d(latest[5]) if latest else 'NA'}\n- foreground ratio checkpoint evidence：{snap['checkpoint_foreground']}\n- best checkpoint：{snap['ck']['best.ckpt']}\n- last checkpoint：{snap['ck']['last.ckpt']}\n- training PID：{st.get('training_pid')}\n- controller PID：{st.get('controller_pid')}\n- reporter PID：{os.getpid()}\n- GPU compute PID：{snap['gpu_process']}\n- GPU memory / utilization：{snap['gpu']}\n- heartbeat：{'ACTIVE' if st.get('training_alive') and st.get('controller_alive') else 'FAILED'}\n- duplicate training：{st.get('duplicate_training')}\n- duplicate controller：{st.get('duplicate_controller')}\n- NaN / Inf / OOM / Traceback / CUDA：{snap['errors'] or 'NONE'}\n- 最近趋势：{'UP' if latest and prior and latest[2]>prior[2] else 'DOWN' if latest and prior else 'NA'}\n- 下一 validation：{((snap['cur']//200)+1)*200}\n- 下一动作：继续自然训练",flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--max-iterations',type=int,default=30000);p.add_argument('--heartbeat-seconds',type=int,default=600);p.add_argument('--attempt1-log');p.add_argument('--pretrain-val',type=float);a=p.parse_args();root=Path(a.run_root).resolve();snap=snapshot(root,a.max_iterations,attempt1_log=a.attempt1_log,pretrain_val=a.pretrain_val);last_val=snap['vals'][-1][0] if snap['vals'] else -1;last_hb=0.;previous_iter=None
    print(f"PERSISTENT_TMUX_REPORTER_ACTIVE pid={os.getpid()} run_root={root} heartbeat={a.heartbeat_seconds}s",flush=True);print_heartbeat(snap,previous_iter);last_hb=time.time();previous_iter=snap['cur']
    while True:
        snap=snapshot(root,a.max_iterations,previous_iter,a.attempt1_log,a.pretrain_val)
        for row in snap['vals']:
            if row[0]>last_val:
                prior=next((x for x in reversed(snap['vals']) if x[0]<row[0]),None);print_validation(row,prior,snap['best'],snap);last_val=row[0]
        if time.time()-last_hb>=a.heartbeat_seconds:print_heartbeat(snap,previous_iter);previous_iter=snap['cur'];last_hb=time.time()
        if not snap['state'].get('training_alive',False):print(f"\nRUN1_{'NATURAL_COMPLETION' if 'NATURAL_COMPLETION' in snap['text'] else 'TRAINING_STOPPED'} {stamp()}",flush=True);break
        time.sleep(5)
    return 0
if __name__=='__main__':raise SystemExit(main())
