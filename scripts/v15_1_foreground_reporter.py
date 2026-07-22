#!/usr/bin/env python3
"""Foreground event stream for user-visible V15.1 monitoring; read-only."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from datetime import datetime
from pathlib import Path

VAL = re.compile(r"VALIDATION iteration=(\d+) selected=(\w+) mDice=([0-9.]+) model1=([0-9.]+) model2=([0-9.]+) RV=([0-9.]+) MYO=([0-9.]+) LV=([0-9.]+)")
LOSS = re.compile(r"iteration (\d+) loss ([0-9.eE+-]+) model1 ([0-9.eE+-]+) model2 ([0-9.eE+-]+)")
ERROR = re.compile(r"NaN|\bInf\b|OOM|out of memory|Traceback|CUDA error|non-finite", re.I)

def alive(pid):
    try: os.kill(pid, 0); return True
    except (ProcessLookupError, PermissionError): return False

def gpu():
    result=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True)
    apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],capture_output=True,text=True)
    return {"gpu":result.stdout.strip(),"compute_apps":apps.stdout.strip().splitlines()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run_root',required=True); p.add_argument('--training_pid',type=int,required=True); p.add_argument('--controller_pid',type=int,required=True); p.add_argument('--start_validation',type=int,default=-1); p.add_argument('--heartbeat_seconds',type=int,default=600); a=p.parse_args()
    root=Path(a.run_root); last_val=a.start_validation; last_heartbeat=0.; prior_heartbeat_iter=None; reported_errors=set()
    print(json.dumps({"event":"REPORTER_ACTIVE","pid":os.getpid(),"run_root":str(root.resolve()),"start_validation":last_val,"heartbeat_seconds":a.heartbeat_seconds}),flush=True)
    while True:
        now=time.time(); console=(root/'console.log').read_text(errors='replace') if (root/'console.log').exists() else ''
        vals=[(int(i),float(v),sel,float(m1),float(m2),float(rv),float(myo),float(lv)) for i,sel,v,m1,m2,rv,myo,lv in VAL.findall(console)]
        losses=[(int(i),float(loss),float(m1),float(m2)) for i,loss,m1,m2 in LOSS.findall(console)]
        best=max(vals,key=lambda row:row[1]) if vals else None
        for row in vals:
            if row[0] > last_val:
                previous=next((item for item in reversed(vals) if item[0] < row[0]),None)
                print(json.dumps({"event":"NEW_VALIDATION","time":datetime.now().astimezone().isoformat(),"iteration":row[0],"mDice":row[1],"selected":row[2],"model1":row[3],"model2":row[4],"RV":row[5],"MYO":row[6],"LV":row[7],"previous":list(previous[:2]) if previous else None,"best":list(best[:2]),"new_best":best[0]==row[0],"training_alive":alive(a.training_pid),"controller_alive":alive(a.controller_pid)}),flush=True)
                last_val=row[0]
        errors=set(match.group(0) for match in ERROR.finditer(console))
        unseen=errors-reported_errors
        if unseen:
            print(json.dumps({"event":"ERROR","errors":sorted(unseen),"iteration":losses[-1][0] if losses else 0}),flush=True); reported_errors |= unseen
        if now-last_heartbeat >= a.heartbeat_seconds:
            iteration=losses[-1][0] if losses else 0; checkpoint_dir=root/'checkpoints'
            checkpoints={path.name:{"bytes":path.stat().st_size,"mtime":datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()} for path in checkpoint_dir.glob('*.ckpt')}
            print(json.dumps({"event":"TEN_MIN_HEARTBEAT","time":datetime.now().astimezone().isoformat(),"iteration":iteration,"iteration_from":prior_heartbeat_iter,"iteration_delta":None if prior_heartbeat_iter is None else iteration-prior_heartbeat_iter,"completion_percent":iteration/30000*100,"latest_loss":list(losses[-1]) if losses else None,"latest_validation":list(vals[-1][:2]) if vals else None,"previous_validation":list(vals[-2][:2]) if len(vals)>1 else None,"best":list(best[:2]) if best else None,"training_pid":a.training_pid,"training_alive":alive(a.training_pid),"controller_pid":a.controller_pid,"controller_alive":alive(a.controller_pid),"reporter_pid":os.getpid(),"gpu":gpu(),"console_mtime":datetime.fromtimestamp((root/'console.log').stat().st_mtime).astimezone().isoformat(),"monitor_mtime":datetime.fromtimestamp((root/'monitor_state.json').stat().st_mtime).astimezone().isoformat(),"checkpoints":checkpoints,"errors":sorted(errors)}),flush=True)
            prior_heartbeat_iter=iteration; last_heartbeat=now
        if not alive(a.training_pid):
            event='NATURAL_COMPLETION' if 'NATURAL_COMPLETION' in console else 'TRAINING_STOPPED'
            print(json.dumps({"event":event,"time":datetime.now().astimezone().isoformat(),"last_iteration":losses[-1][0] if losses else 0,"last_validation":list(vals[-1][:2]) if vals else None,"best":list(best[:2]) if best else None}),flush=True); break
        time.sleep(5)
    return 0
if __name__=='__main__': raise SystemExit(main())
