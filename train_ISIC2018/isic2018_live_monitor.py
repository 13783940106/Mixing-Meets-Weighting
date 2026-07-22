#!/usr/bin/env python3
"""Foreground tmux monitor for the single persistent ISIC Controller."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; C=ROOT/'runs/ISIC2018/control'
def load(name):
 try:return json.loads((C/name).read_text())
 except Exception:return {}
def alive(pid):
 try:os.kill(int(pid),0);return True
 except Exception:return False
def latest_run_from_state(s):
 artifact=str(s.get('current_artifact') or '')
 for part in artifact.split(','):
  p=ROOT/part
  if p.name=='result.json':
   return p.parent
 manifest=ROOT/'audit/isic2018_candidate_full30k_source_manifest.json'
 if manifest.exists():
  try:
   data=json.loads(manifest.read_text())
   state=s.get('current_state','')
   ratio='5p' if '5P' in state else '10p' if '10P' in state else None
   if ratio:
    return Path(json.loads((ROOT/data['configs'][ratio]).read_text())['run_root'])
  except Exception:
   return None
 return None
def latest_metrics(run):
 if not run:return None
 p=run/'metrics.jsonl'
 if not p.exists():return None
 last=val=None
 try:
  for line in p.read_text(errors='replace').splitlines():
   if line.strip():
    obj=json.loads(line); last=obj
    if obj.get('kind')=='validation': val=obj
  ckpts=len(list((run/'checkpoints').glob('*.ckpt'))) if (run/'checkpoints').exists() else 0
  return last,val,ckpts
 except Exception:return None
def main():
 while True:
  s,a,h=load('prefull_state.json'),load('prefull_active_run.json'),load('prefull_heartbeat.json'); pid=a.get('child_pid',s.get('child_pid'))
  print('\033[2J\033[H'+time.strftime('%F %T')+'  TMUX_SESSION=isic2018-prefull',flush=True)
  for k in ('current_state','last_completed_state','next_automatic_state','terminal_allowed','controller_pid','child_pid','gpu_pid','last_failure_class'): print(f'{k.upper()}={s.get(k,a.get(k,"NOT_EMITTED_YET"))}',flush=True)
  print(f'CONTROLLER_ALIVE={alive(s.get("controller_pid"))} CHILD_ALIVE={alive(pid)} HEARTBEAT={h.get("heartbeat_timestamp","NOT_EMITTED_YET")}',flush=True)
  run=latest_run_from_state(s); lm=latest_metrics(run)
  if run: print(f'RUN_ROOT={run}',flush=True)
  if lm:
   last,val,ckpts=lm
   if last: print(f'LATEST_UPDATE={last.get("iteration")} LOSS={last.get("loss")} CANDIDATE_ONLY_LOSS={last.get("candidate_only_loss")}',flush=True)
   if val: print(f'LATEST_VAL_ITER={val.get("iteration")} VAL_DICE={val.get("mean_image_dice")} VAL_IOU={val.get("mean_image_iou")}',flush=True)
   print(f'CHECKPOINT_COUNT={ckpts}',flush=True)
  if run and (run/'result.json').exists():
   try:
    r=json.loads((run/'result.json').read_text())
    print(f'BEST_DICE={r.get("best_dice")} BEST_IOU={r.get("best_iou")} BEST_ITER={r.get("best_iteration")} COUNTERS={r.get("counters")}',flush=True)
   except Exception: pass
  log=a.get('stdout_path')
  if log and Path(log).exists(): print(Path(log).read_text(errors='replace').splitlines()[-1],flush=True)
  time.sleep(5)
if __name__=='__main__':main()
