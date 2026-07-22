#!/usr/bin/env python3
"""Persistent, evidence-first ISIC pre-Full task queue entrypoint."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CONTROL=ROOT/'runs/ISIC2018/control'; PY=sys.executable
QUEUE=[
 ('S00_VERIFY_EXISTING_ARTIFACTS','verify frozen SHA and active artifacts','S01_BASELINE_RESUME_COMPARE'),
 ('S01_BASELINE_RESUME_COMPARE','compare strict reference/interrupted/resumed trace','S02_CANDIDATE_RESUME_RUN'),
 ('S02_CANDIDATE_RESUME_RUN','execute candidate strict 3+3 resume','S03_CANDIDATE_RESUME_COMPARE'),
 ('S03_CANDIDATE_RESUME_COMPARE','compare candidate trace','S04_RESUME_NEGATIVE_GATES'),
 ('S04_RESUME_NEGATIVE_GATES','run negative resume gates','S05_PARENT_AND_CANDIDATE_PARITY'),
 ('S05_PARENT_AND_CANDIDATE_PARITY','run parity and candidate-off matrix','S06_CHECKPOINT_INFRASTRUCTURE'),
 ('S06_CHECKPOINT_INFRASTRUCTURE','validate checkpoint and manifest','S07_CONTROLLER_LIFECYCLE_TESTS'),
 ('S07_CONTROLLER_LIFECYCLE_TESTS','run controller lifecycle matrix','S08_FULL_UNIT_TEST_MATRIX'),
 ('S08_FULL_UNIT_TEST_MATRIX','run full unit matrix','S09_AFFECTED_PREFLIGHT'),
 ('S09_AFFECTED_PREFLIGHT','run affected preflight','S10_PROTOCOL_AND_BUDGET'),
 ('S10_PROTOCOL_AND_BUDGET','freeze protocol and budget','S11_FORMAL_SOURCE_FREEZE'),
 ('S11_FORMAL_SOURCE_FREEZE','freeze formal source','S12_FORMAL_RESOURCE_PROBE'),
 ('S12_FORMAL_RESOURCE_PROBE','formal resource probe','S13_FORMAL_SMOKE_5P_BASELINE'),
 ('S13_FORMAL_SMOKE_5P_BASELINE','formal smoke','S14_FORMAL_SMOKE_5P_CANDIDATE'),
 ('S14_FORMAL_SMOKE_5P_CANDIDATE','formal smoke','S15_FORMAL_SMOKE_10P_BASELINE'),
 ('S15_FORMAL_SMOKE_10P_BASELINE','formal smoke','S16_FORMAL_SMOKE_10P_CANDIDATE'),
 ('S16_FORMAL_SMOKE_10P_CANDIDATE','formal smoke','S17_FORMAL_SHORT5K_5P_BASELINE'),
 ('S17_FORMAL_SHORT5K_5P_BASELINE','formal short5k','S18_FORMAL_SHORT5K_5P_CANDIDATE'),
 ('S18_FORMAL_SHORT5K_5P_CANDIDATE','formal short5k','S19_FORMAL_SHORT5K_10P_BASELINE'),
 ('S19_FORMAL_SHORT5K_10P_BASELINE','formal short5k','S20_FORMAL_SHORT5K_10P_CANDIDATE'),
 ('S20_FORMAL_SHORT5K_10P_CANDIDATE','formal short5k','S21_CHECKPOINT_FINAL_AUDIT'),
 ('S21_CHECKPOINT_FINAL_AUDIT','final checkpoint audit','S22_FULL_PACKAGE_DRY_RUN'),
 ('S22_FULL_PACKAGE_DRY_RUN','generate/dry-run Full package','S23_PREFULL_MASTER_AUDIT'),
 ('S23_PREFULL_MASTER_AUDIT','master audit','S24_STOP_FULL_NOT_STARTED')]
def write(state):
 CONTROL.mkdir(parents=True,exist_ok=True);(CONTROL/'prefull_state.json').write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');(CONTROL/'prefull_heartbeat.json').write_text(json.dumps(state,indent=2,sort_keys=True)+'\n')
def main():
 # The queue is intentionally durable; execution procedures are appended only
 # after their entry gates are concretely implemented and tested.
 state={'current_state':'S00_VERIFY_EXISTING_ARTIFACTS','next_automatic_state':'S01_BASELINE_RESUME_COMPARE','terminal_allowed':False,'terminal_reason':'INTERMEDIATE_STATE','controller_pid':os.getpid(),'child_pid':None,'gpu_pid':None,'queue':[{'state':a,'action':b,'next':c} for a,b,c in QUEUE],'heartbeat_timestamp':time.time()}
 write(state)
 subprocess.run([PY,'-m','train_ISIC2018.isic2018_assert_terminal_state'],cwd=ROOT,check=False)
 # Run concrete first gate: cryptographic frozen-artifact verification.
 result=subprocess.run(['bash','-lc','cd audit && sha256sum -c isic2018_v2_1_SHA256SUMS && cd /home/renguanming/datasets/ISIC2018/splits/isic2018_official_5p10p_seed1337_v1 && sha256sum -c SHA256SUMS'],cwd=ROOT,capture_output=True,text=True)
 state.update({'last_completed_state':'S00_VERIFY_EXISTING_ARTIFACTS' if result.returncode==0 else None,'current_state':'S01_BASELINE_RESUME_COMPARE','next_automatic_state':'S02_CANDIDATE_RESUME_RUN','verification_output':result.stdout[-4000:],'heartbeat_timestamp':time.time()});write(state)
 # Keep the server-side controller alive while later state implementations are added.
 while True: state['heartbeat_timestamp']=time.time();write(state);time.sleep(30)
if __name__=='__main__':main()
