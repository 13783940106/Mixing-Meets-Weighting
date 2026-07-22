#!/usr/bin/env python3
"""Executable terminal whitelist for every ISIC pre-Full exit path."""
from __future__ import annotations
import json
from pathlib import Path
from train_ISIC2018.isic2018_prefull_controller import SUCCESS_TERMINAL, LEGAL_BLOCKER_PREFIX, assert_terminal_state_allowed

ROOT=Path(__file__).resolve().parents[1]; CONTROL=ROOT/'runs/ISIC2018/control'
def read(name):
 p=CONTROL/name
 return json.loads(p.read_text()) if p.exists() else {}
def main():
 state=read('prefull_state.json'); active=read('prefull_active_run.json'); heartbeat=read('prefull_heartbeat.json')
 current=state.get('current_state',state.get('state','IMPLEMENT_UNIFIED_TRAIN_CORE'))
 evidence=state.get('blocker_evidence')
 try:
  allowed,reason=assert_terminal_state_allowed(current,evidence)
 except RuntimeError:
  state.update({'terminal_allowed':False,'terminal_reason':'INTERMEDIATE_STATE','next_automatic_state':state.get('next_automatic_state','IMPLEMENT_UNIFIED_TRAIN_CORE'),'early_stop_guard_pass':True,'active_run_observed':active.get('active_run'),'heartbeat_observed':heartbeat.get('heartbeat_timestamp')})
  CONTROL.mkdir(parents=True,exist_ok=True);(CONTROL/'prefull_state.json').write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
  raise SystemExit(73)
 state.update({'terminal_allowed':allowed,'terminal_reason':reason,'early_stop_guard_pass':True})
 (CONTROL/'prefull_state.json').write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
